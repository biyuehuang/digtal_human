from funasr import AutoModel as funasr_AutoModel
import torch
import time
import intel_extension_for_pytorch as ipex


t0 = time.time()
print("-----------loading Long ASR model")# funasr==1.0.11
funasr_model = funasr_AutoModel(model="./models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", model_revision="v2.0.4",
                # vad_model="../rag/models/speech_fsmn_vad", vad_model_revision="v2.0.4",
                #punc_model="./punc_ct-transformer_cn-en-common-vocab471067-large", punc_model_revision="v2.0.4",
                punc_model="../rag/models/punc_ct-transformer", punc_model_revision="v2.0.4",
                # spk_model="cam++", spk_model_revision="v2.0.2",
                device="cpu",# cpu 可改成xpu
                )
print("-----------loading Long ASR model---Done")
t1 = time.time()
print("-------load time(s): ", t1-t0)

wav_file = ["05s.wav", "05s_chengyu.wav", "10s_asr_example.wav","10s_chinese_16k.wav","60s.wav","60s_vad_example.wav"]
avg_t = 0
for i in range(len(wav_file)):
    file = wav_file[i]
    print("--------------file: ", file)
    tt = 0
    N = 6
    for n in range(N):
        t0 = time.time()
        with torch.inference_mode():
            res = funasr_model.generate(input="./test_audio/"+file, batch_size_s=1)#16， BS可根据实际需求设置
            # res = self.funasr_model.generate(input=file, batch_size_s=8, hotword="./models/speech_seaco_paraformer_large_asr_nat/hotword.txt")
            text = res[0]['text']
        t1 = time.time()
        print("----------ASR latency(s): ", t1-t0)
        if n != 0:
            tt += t1-t0
    print("=======avg: ", tt/(N-1))
    avg_t += tt/(N-1)
    if (i+1)%2 == 0:
        print("*******************", avg_t/2)
        print("\n\n")
        avg_t = 0
