# Bark voice cloning
This repository is a version of https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer, but with improved training, leading to a higher quality Hubert quantizer for [suno/bark](https://github.com/suno-ai/bark). This leads to higher quality and more consistent voice cloning with Bark. 
## Please read
This code works on python 3.10, i have not tested it on other versions. Some older versions will have issues.

## Voice cloning with bark in high quality?
It's possible now.

https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer/assets/36931363/516375e2-d699-44fe-a928-cd0411982049



## How do I clone a voice?
For developers:
* [code examples on huggingface model page](https://huggingface.co/GitMylo/bark-voice-cloning)

For everyone:
* [audio-webui with bark and voice cloning](https://github.com/gitmylo/audio-webui)
* [online huggingface voice cloning space](https://huggingface.co/spaces/GitMylo/bark-voice-cloning)
* [interactive python notebook](notebook.ipynb)

## Voices cloned aren't very convincing, why are other people's cloned voices better than mine?
Make sure these things are **NOT** in your voice input: (in no particular order)
* Noise (You can use a noise remover before)
* Music (There are also music remover tools) (Unless you want music in the background)
* A cut-off at the end (This will cause it to try and continue on the generation)
* Under 1 second of training data (i personally suggest around 10 seconds for good potential, but i've had great results with 5 seconds as well.)

What makes for good prompt audio? (in no particular order)
* Clearly spoken
* No weird background noises
* Only one speaker
* Audio which ends after a sentence ends
* Regular/common voice (They usually have more success, it's still capable of cloning complex voices, but not as good at it)
* Around 10 seconds of data

## Pretrained models
### Official

| Name                                                                                                                                         | HuBERT Model                                                              | Quantizer Version | Epoch | Language | Dataset                                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|-------------------|-------|----------|--------------------------------------------------------------------------------------------------|
| [quantifier_hubert_base_ls960.pth](https://huggingface.co/GitMylo/bark-voice-cloning/blob/main/quantifier_hubert_base_ls960.pth)             | [HuBERT Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) | 0                 | 3     | ENG      | [GitMylo/bark-semantic-training](https://huggingface.co/datasets/GitMylo/bark-semantic-training) |
| [quantifier_hubert_base_ls960_14.pth](https://huggingface.co/GitMylo/bark-voice-cloning/blob/main/quantifier_hubert_base_ls960_14.pth)       | [HuBERT Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) | 0                 | 14    | ENG      | [GitMylo/bark-semantic-training](https://huggingface.co/datasets/GitMylo/bark-semantic-training) |
| [quantifier_V1_hubert_base_ls960_23.pth](https://huggingface.co/GitMylo/bark-voice-cloning/blob/main/quantifier_V1_hubert_base_ls960_23.pth) | [HuBERT Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) | 1                 | 23    | ENG      | [GitMylo/bark-semantic-training](https://huggingface.co/datasets/GitMylo/bark-semantic-training) |

### Community

| Author                                      | Name                                                                                                                                                                 | HuBERT Model                                                              | Quantizer Version | Epoch | Language | Dataset                                                                                                                      |
|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|-------------------|-------|----------|------------------------------------------------------------------------------------------------------------------------------|
| [HobisPL](https://github.com/HobisPL)       | [polish-HuBERT-quantizer_8_epoch.pth](https://huggingface.co/Hobis/bark-voice-cloning-polish-HuBERT-quantizer/blob/main/polish-HuBERT-quantizer_8_epoch.pth)         | [HuBERT Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) | 1                 | 8     | POL      | [Hobis/bark-polish-semantic-wav-training](https://huggingface.co/datasets/Hobis/bark-polish-semantic-wav-training)           |
| [C0untFloyd](https://github.com/C0untFloyd) | [ german-HuBERT-quantizer_14_epoch.pth](https://huggingface.co/CountFloyd/bark-voice-cloning-german-HuBERT-quantizer/blob/main/german-HuBERT-quantizer_14_epoch.pth) | [HuBERT Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) | 1                 | 14    | GER      | [CountFloyd/bark-german-semantic-wav-training](https://huggingface.co/datasets/CountFloyd/bark-german-semantic-wav-training) |


## For developers: Implementing voice cloning in your bark projects
* Simply copy the files from [this directory](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer/tree/master/hubert) into your project.
* The [hubert manager](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer/blob/master/hubert/hubert_manager.py) contains methods to download HuBERT and the custom Quantizer model.
* Loading the [CustomHuBERT](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer/blob/master/hubert/pre_kmeans_hubert.py) should be pretty straightforward
* The [notebook](notebook.ipynb) contains code to use on cuda or cpu. Instead of just cpu.
```python
from hubert.pre_kmeans_hubert import CustomHubert
import torchaudio

# Load the HuBERT model,
# checkpoint_path should work fine with data/models/hubert/hubert.pt for the default config
hubert_model = CustomHubert(checkpoint_path='path/to/checkpoint')

# Run the model to extract semantic features from an audio file, where wav is your audio file
wav, sr = torchaudio.load('path/to/wav') # This is where you load your wav, with soundfile or torchaudio for example

if wav.shape[0] == 2:  # Stereo to mono if needed
    wav = wav.mean(0, keepdim=True)

semantic_vectors = hubert_model.forward(wav, input_sample_hz=sr)
```
* Loading and running the [custom kmeans](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer)

```python
import torch
from hubert.customtokenizer import CustomTokenizer

# Load the CustomTokenizer model from a checkpoint
# With default config, you can use the pretrained model from huggingface
# With the default setup from HuBERTManager, this will be in data/models/hubert/tokenizer.pth
tokenizer = CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer.pth')  # Automatically uses the right layers

# Process the semantic vectors from the previous HuBERT run (This works in batches, so you can send the entire HuBERT output)
semantic_tokens = tokenizer.get_token(semantic_vectors)

# Congratulations! You now have semantic tokens which can be used inside of a speaker prompt file.
```

## How do I train it myself?
Simply run the training commands.

A simple way to create semantic data and wavs for training, is with my script: [bark-data-gen](https://github.com/gitmylo/bark-data-gen). But remember that the creation of the wavs will take around the same time if not longer than the creation of the semantics. This can take a while to generate because of that.

For example, if you have a dataset with zips containing audio files, one zip for semantics, and one for the wav files. Inside of a folder called "Literature"

You should run `process.py --path Literature --mode prepare` for extracting all the data to one directory

You should run `process.py --path Literature --mode prepare2` for creating HuBERT semantic vectors, ready for training

You should run `process.py --path Literature --mode train` for training

And when your model has trained enough, you can run `process.py --path Literature --mode test` to test the latest model.

## Disclaimer
I am not responsible for audio generated using semantics created by this model. Just don't use it for illegal purposes.
