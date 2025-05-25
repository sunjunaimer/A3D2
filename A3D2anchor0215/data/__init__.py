import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import random
import pandas as pd
import os
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as transforms
from argparse import Namespace
import librosa
from transformers import BertTokenizerFast, ViTImageProcessor, RobertaTokenizer
import json
import torch.nn.functional as F
from pydub import AudioSegment
from pydub.generators import WhiteNoise
import tempfile
import torchaudio

seed = 42 #100 #42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

with open('config_dir.json', 'r') as file:
    cfg = json.load(file)


wav_dir= cfg['wav2vec']
bert_dir = cfg['bert-uncased']
vit_dir = cfg['vit']



vision_model = cfg['vision']['model']





os.environ["TOKENIZERS_PARALLELISM"] = "false"


emotion_labels = ['neutral', 'happy', 'sad', 'angry']

mintrec_lables = ['Inform', 'Introduce', 'Oppose', 'Leave', 'Complain', 'Care', 'Praise', 'Arrange', 'Comfort', 'Greet', 'Agree', 'Apologise', 'Advise', 'Criticize', 'Thank', 'Taunt', 'Prevent', 'Flaunt', 'Ask for help', 'Joke']






class MultiModalDataset(Dataset):  
    def __init__(self, opt, dataset_name, mode):
        """
        annotations_file (string): 
        img_dir, audio_dir, text_dir (string): 
        transform (callable, optional): 
        target_transform (allable, optional): 
        """
        dataset_name = dataset_name.lower()
        self.dataset_dir = os.path.join(opt.datasets_folder, dataset_name)
        if opt.domain == 'source':
            csv_name = 'label_s.csv'
        else:
            csv_name = 'label_t.csv'
        
        self.label_df = pd.read_csv(
            os.path.join(self.dataset_dir, csv_name),
            dtype={'video_id': str, 'clip_id': str, 'text': str}
        )
        print(f'reading csv file: {csv_name}')
        self.mode = mode
        self.label_df = self.label_df[self.label_df['mode'] == self.mode]
        self.dataset_name = dataset_name

        
        self.au_audio, self.au_vision, self.no_vision, self.au_text = map(float, opt.tmp_weight.split(' '))

        if cfg['vision']['model'] == 'ViT':
            self.image_premodel = ViTImageProcessor.from_pretrained(vit_dir)
            

    def __len__(self):
        return len(self.label_df)

    def get_dir(self, idx):
        video_id = self.label_df.iloc[idx]['video_id']
        clip_id = self.label_df.iloc[idx]['clip_id']
        self.video_id = video_id
        self.clip_id = clip_id

       
        image_folder = str(video_id) + '_' + str(clip_id)
        img_path = Path(self.dataset_dir) / image_folder
        wav_name = str(video_id) + '_' + str(clip_id) + '.wav'
        wav_path = Path(self.dataset_dir) / wav_name
        return (img_path, wav_path)


    def random_mask(self, words, p, seed=42):
 
        random.seed(seed)
        #obviously, if there's only one word, don't delete it
        if len(words) == 1:
            return words

        #randomly delete words with probability p
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r <= p:
                new_words.append('MASK')
            if r > p:
                new_words.append(word)

        return new_words

    def random_deletion(self, words, p, seed=42):

        random.seed(seed)
        #obviously, if there's only one word, don't delete it
        if len(words) == 1:
            return words

        #randomly delete words with probability p
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)

        #if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return [words[rand_int]]
        
        if len(new_words) == len(words) and len(new_words) > 1:
            rand_int = random.randint(0, len(words)-1)
            return [words[i] for i in range(len(words)) if i != rand_int]

        return new_words



    def add_noise(self, audio, p=1):


        audio = AudioSegment.from_file(audio)

        noise = WhiteNoise().to_audio_segment(duration=len(audio))
      
        noise = noise + audio.dBFS - noise.dBFS + (p-1)*20 # 
        noisy_audio = audio.overlay(noise)

        if not os.path.exists(os.path.join(self.dataset_dir, 'tmp')):
            os.makedirs(os.path.join(self.dataset_dir, 'tmp'))
        wav_name = self.video_id + '_' + self.clip_id + '.wav'
        wav_dir =  os.path.join(self.dataset_dir, 'tmp', wav_name)
        noisy_audio.export(wav_dir, format="wav")

        return wav_dir
        


    def preprocess_image(self, image_path):  
        image_files = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.jpg')])
        images = []
        num_images = 10
        num = min(len(image_files), num_images)  #10

        # selected_images = random.sample(image_files, num)
        # selected_images_sorted = sorted(selected_images, key=image_files.index)

        # Select images uniformly based on their indices
        if num > 1:
            interval = len(image_files) / num
            selected_indices = [int(i * interval) for i in range(num)]
            selected_images = [image_files[idx] for idx in selected_indices]
        else:
            selected_images = [image_files[0]] if image_files else []

        selected_images_sorted = sorted(selected_images, key=image_files.index)

        
        if vision_model == 'APViT':
            for i, image_file in enumerate(selected_images_sorted):

                img = Image.open(image_file)
                if self.au_vision != 1:
                    enhancer = ImageEnhance.Brightness(img)
                    brightness_factor = self.au_vision  
                    img = enhancer.enhance(brightness_factor)

                preprocess = transforms.Compose([
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                img = preprocess(img)

                
                if self.no_vision > 0:
                    mean = 0.0
                    std_dev = self.no_vision 
                    gaussian_noise = torch.normal(mean, std_dev, size=img.shape)
                    img = img + gaussian_noise
                images.append(img)

            image_size = [112, 112]
            if len(images) < num_images:
                zero_image = torch.zeros((3, image_size[0], image_size[1]))
                while len(images) < num_images:
                    images.append(zero_image)

            images = torch.stack(images, dim=0)
            # print(f'****{image.shape}')
            return images, num
        

        elif vision_model == 'ViT':
            video_image = []
            for i, image_file in enumerate(selected_images_sorted):
            
                img = Image.open(image_file)
                if self.au_vision != 1:
                    enhancer = ImageEnhance.Brightness(img)
                    brightness_factor = self.au_vision  #
                    img = enhancer.enhance(brightness_factor)

                video_image.append(img)

            images = self.image_premodel(images=video_image, return_tensors="pt").pixel_values           
            return images, num


    def preprocess_audio(self, file):
 
        # Desired sample rate
        desired_sample_rate = 16000  # You can change this to your desired sample rate

        # 1. Load the audio with the original sample rate
        waveform, original_sample_rate = torchaudio.load(file)
        waveform = waveform.mean(dim=0)

        # 2. Resample the audio if the original sample rate is different from the desired one
        if original_sample_rate != desired_sample_rate:
            resample_transform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=desired_sample_rate)
            waveform = resample_transform(waveform)

        if self.au_audio != -1:
            noise = torch.randn_like(waveform) * self.au_audio  # Generate Gaussian noise

            # 4. Add noise to the original waveform
            waveform = waveform + noise

            # 5. Normalize the noisy waveform (optional)
            waveform = torch.clip(waveform, -1.0, 1.0) 
        waveform = waveform.numpy()

        return waveform

    def preprocess_text(self, sentence, p_text):
        if p_text == 0:
            return sentence
        words = sentence.split(' ')
        words = [word for word in words if word != '']
        #a_words = self.random_deletion(words, p_text)
        a_words = self.random_mask(words, p_text)
        new_sentence = ' '.join(a_words)
        return new_sentence



    def __getitem__(self, idx):

        img_path, wav_path = self.get_dir(idx)

        text = '[CLS]' + self.preprocess_text(self.label_df.iloc[idx]['text'], self.au_text)
        label = self.label_df.iloc[idx]['emotion']
        if self.dataset_name != 'mintrec':
            label = emotion_labels.index(label)  
        else:
            label = mintrec_lables.index(label)       
        
        image, num = self.preprocess_image(img_path)
        audio = self.preprocess_audio(wav_path)

        sample = {'image': image, 'audio': audio, 'text': text, 'label': label}

        return sample





class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, dataset_name, mode):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.dataset = MultiModalDataset(opt, dataset_name, mode)

        if cfg['text']['model'] == 'bert-uncased':
            self.text_premodel = BertTokenizerFast.from_pretrained(cfg['bert-uncased'])
        
        if cfg['text']['model'] == 'roberta-base':
            self.text_premodel = RobertaTokenizer.from_pretrained(cfg['roberta-base'])
        
        if cfg['text']['model'] == 'roberta-base-emo':
            merges_file_path = "/path/to/roberta-base-emo-merges.txt" 
            merges_file_path = cfg['bert-uncased'] + '/merges.txt'
            vocabulary_file_path = cfg['bert-uncased'] + '/vocab.json'
            self.text_premodel = RobertaTokenizer.from_pretrained(cfg['roberta-base-emo'])
        
        self.audio_model = cfg['audio']['model']


        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True,  
            pin_memory=True,
            num_workers=2,
            prefetch_factor=2,
            drop_last=True,
            collate_fn=self.collate_fn,
        )


    def collate_fn(self, batch):
        images = [item['image'] for item in batch]  #
        audios = [item['audio'] for item in batch]  
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch])

        duration = torch.tensor([int(len(item['audio']) / 320)for item in batch])
        
        audio_final = {'audio':audios, 'duration':duration}
        encoded_text = self.text_premodel(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        if cfg['text']['model'] == 'bert-uncased':
            encoded_text = {'input_ids': encoded_text['input_ids'],
                    'token_type_ids': encoded_text['token_type_ids'],
                    'attention_mask': encoded_text['attention_mask']}     

        if cfg['text']['model'] in ['roberta-base-emo', 'roberta-bae']:
            encoded_text = {'input_ids': encoded_text['input_ids'],
                    'attention_mask': encoded_text['attention_mask']}       

        return {'vision': images, 'audio': audio_final, 'text': encoded_text, 'label': labels}

            

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):  
            yield data





def create_dataset_with_args(opt, dataset_name, set_name):

    dataloaders = tuple(CustomDatasetDataLoader(opt, dataset_name, mode) for mode in set_name)

    return dataloaders if len(dataloaders) > 1 else dataloaders[0]





