from flask import Flask, request, jsonify
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import subprocess
import requests
from wtpsplit import SaT
import bleach

app = Flask(__name__)

BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None

def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()

    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations
def upload_to_cms(file_obj, file_name, description, token):
    url = "https://ig.gov-cloud.ai/mobius-content-service/v1.0/content/upload"
    headers = {
        'Authorization': f'Bearer {token}'
    }
    files = {
        'file': (file_name, file_obj, 'audio/wave')
    }
    file_naming = file_name
    file_name_without_extension = file_naming.removesuffix(".wav")
    params = {
        'filePath': '/bottle/limka/soda/',
        'file name without extension': file_name_without_extension,
        'filePathAccess': 'public',
        'description': description,
        'overrideFile': 'false'
    }
    response = requests.post(url, headers=headers, files=files, params=params)
    return response.json()

@app.route('/translate', methods=['POST'])
def translate():
    if 'file' not in request.files or 'token' not in request.form:
        return "Missing file or token", 400
    file = request.files['file']
    token = request.form['token']
    if file.filename == '':
        return "No selected file", 400
    input_text = file.read().decode('utf-8')
    # Sanitize input_text to prevent XSS
    input_text = bleach.clean(input_text)
    # Initialize the SaT model
    character_count1 = len(input_text)
    print(f"Character count of input: {character_count1}")
    sat_sm = SaT("sat-12l")
    # sat_sm.half().to("cuda")  # optional, see above
    en_sents = sat_sm.split(input_text)
    en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
    en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic", quantization)
    ip = IndicProcessor(inference=True)
    src_lang, tgt_lang = "eng_Latn", "guj_Gujr"
    guj_translations = batch_translate(en_sents, src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, ip)
    output_text = '\n'.join(guj_translations)
    file_path = 'translations.txt'

    # Open the file in write mode and store the output_text
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(output_text)

    character_count2 = len(output_text)
    print(f"Text has been written to {file_path}")
    print(f"Character count of translated: {character_count2}")
    del en_indic_tokenizer, en_indic_model
    try:
        # Run subprocess for Gujarati text synthesis
        gujarati_process = subprocess.Popen([
            "python3", "-m", "TTS.bin.synthesize",
            "--text", output_text,
            "--model_path", "/Users/prasanth/PycharmProjects/APITTS/gu/fastpitch/best_model.pth",
            "--config_path", "/Users/prasanth/PycharmProjects/APITTS/gu/fastpitch/config.json",
            "--vocoder_path", "/Users/prasanth/PycharmProjects/APITTS/gu/hifigan/best_model.pth",
            "--vocoder_config_path", "/Users/prasanth/PycharmProjects/APITTS/gu/hifigan/config.json",
            "--out_path", "outputguj.wav",
            "--speaker_idx", "female"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gujarati_stdout, gujarati_stderr = gujarati_process.communicate()
        if gujarati_process.returncode != 0:
            return gujarati_stderr.decode('utf-8'), 500
        # Upload to CMS
        with open("outputguj.wav", "rb") as gujarati_file:
            gujarati_response = upload_to_cms(gujarati_file, "outputguj.wav", "Gujarati audio file", token)
        if 'cdnUrl' not in gujarati_response:
            return jsonify({
                "error": "Error in CMS upload response",
                "gujarati_response": gujarati_response
            }), 500
        gujarati_cdn_url = gujarati_response['cdnUrl']
        gujarati_url = f"https://cdn.gov-cloud.ai{gujarati_cdn_url}"
        os.remove("outputguj.wav")
        return jsonify({
            "gujarati_audio": gujarati_url
        })
    except subprocess.CalledProcessError as e:
        return str(e), 500

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    app.run(host='0.0.0.0', port=8080)
