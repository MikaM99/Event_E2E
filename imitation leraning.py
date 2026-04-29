from __future__ import print_function

import os

import scipy

#import tensorflow as tf
import torch
import numpy as np

#slim = tf.contrib.slim

#from carla.agent import Agent
#from carla.carla_server_pb2 import Control
#from agents.imitation.imitation_learning_network import load_imitation_learning_network
from agents.imitation.pytorch_imitation_learning_network import Mymodel

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

class DDD17Dataset(Dataset):

    def __init__(self, images, events, meta):
        self.images = images
        self.events = events
        self.meta = meta

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].float()
        event = self.events[idx].float()
        target = self.meta[idx].float()
        return image, event, target

# CARLA Server
# Terminal mit conda E2E
# go to /home/carlas/Simulators/CARLA_0.9.16
# ./CarlaUE4.sh


class ImitationLearning():#Agent):

    def __init__(self, city_name, avoid_stopping, raw_data, memory_fraction=0.25, image_cut=[115, 510]):
        #load model aufrufen um checkpoint aufzurufen

        self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        torch.cuda.empty_cache()

        self._image_size = (260, 346, 3)


        #self._input_images = torch.Tensor(raw_data[0]).float().unsqueeze(0).to(device) 
        #in load data mit übernehmen
        #print(type(raw_data))
        #print(len(raw_data))
        #print(raw_data[0][0].shape)
        #print(self._input_images.shape)
        #self._input_data = torch.Tensor(raw_data[2]) # ["vehicle_speed", "steering"]

        self._input_events = raw_data[1]

        #with tf.name_scope("Network"):
        #self._network_tensor = load_imitation_learning_network(self._input_images,#anstelle load imitation forward aufrufen in 
        #                                                           self._input_data,
        #                                                           self._image_size, self._dout)
        USE_FRN = True

        self.model = Mymodel(image_shape=self._image_size, Device=device, use_fusion=USE_FRN)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4) #geändert
        self.criterion = torch.nn.MSELoss()
        self.dataset = DDD17Dataset (raw_data[0], raw_data[1], raw_data[2])
        self.dataloader = DataLoader(self.dataset, batch_size=8, shuffle=True, num_workers=0)
        sample = self.dataset[0]
        print("dataset sample len:", len(sample))
        print("dataset sample types:", type(sample), type(sample[0]), type(sample[1]))
        import os
        dir_path = os.path.dirname(__file__)

        self._models_path = dir_path + '/model/'
        os.makedirs(self._models_path, exist_ok=True)

        # wird dann in train und test jeweils aufgerufen 
        self.train(raw_data, device, 5)


    def load_model(self):

        #variables_to_restore = tf.global_variables()

        #saver = tf.train.Saver(variables_to_restore, max_to_keep=0)

        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path')
        
        checkpoint = torch.load(self._models_path, weights_only=True)
        self._network_tensor.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        #ckpt = tf.train.get_checkpoint_state(self._models_path)
        #if ckpt:
        #    print('Restoring from ', ckpt.model_checkpoint_path)
        #    saver.restore(self._sess, ckpt.model_checkpoint_path)
        #else:
        #    ckpt = 0

        return checkpoint

    #def _compute_action(self, rgb_image, speed_steer):

        #print(torch.max(rgb_image))
        #image_input = rgb_image / 255.0
        #print(torch.max(image_input))

        #predicted_speed, predicted_steer = self.model.forward(input_data=speed_steer,input_image=image_input)

        #control = {}
        #control["steer"] = predicted_steer
        #control["speed"] = predicted_speed

        #return control
    
    def train(self, raw_data, device, num_epochs=1):
        self.model.train()
        #for in_image, in_data in zip(raw_data[0][:10], raw_data[2][:10]): # schneller alternative zu zip
        #for i in range(10):  #später für alle Daten: (len(raw_data[0])):
            #in_image = raw_data[0][i]
            #in_data = raw_data[2][i]
            #image = torch.Tensor(in_image).float().unsqueeze(0).to(device)
            #data = torch.Tensor(in_data)
            #print(image.shape, data.shape)
            #output = self._compute_action(image, data)
            #print(output)

        #for i in range(0, 10):

            #in_image = raw_data[0][i]
            #in_data = raw_data[2][i]
            #image = torch.tensor(in_image).float().unsqueeze(0).to(device)
            #target = torch.tensor(in_data).float().to(device)
        best_loss = float("inf")#if bedingung anzahl epochen loss unter schwellwert
        Schwellwert = 20000
        epoch_losses = []
        first_batch = next(iter(self.dataloader))
        print("batch len", len(first_batch))
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for images, events, targets in self.dataloader:
                images = images.to(device)
                targets = targets.to(device)
                events = events.to(device)

                self.optimizer.zero_grad()
                #print(targets.shape)
                speed_input = targets[:, 0].unsqueeze(1)
                steer_target = targets[:, 1].unsqueeze(1)
                output = self.model(images, speed_input, events)
                loss = self.criterion(output, steer_target)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.dataloader)
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, avg_loss: {avg_epoch_loss}")
            if avg_epoch_loss < Schwellwert:
                torch.save(self.model.state_dict(), self._models_path + "/checkpoint.pth")
                print("Checkpoint saved")
                best_loss = avg_epoch_loss
                checkpoint_path = os.path.join(self._models_path, f"best_model_epoch_{epoch+1}.pth")#+epochennummer übergeben f str
                # chechpoint speicher abhänging welche branch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_loss,
                }, checkpoint_path)

                print("Checkpoint saved:", checkpoint_path)
        print("All epoch losses:", epoch_losses)# als array machen nicht als zahl

    def test(self):
        #output = self.model(image, target)
        # criterion / auswertfunktion?
        # mit Ground truth
        #output = self.model(images, events, meta_input)   xxx
        pass
