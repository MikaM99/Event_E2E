from __future__ import print_function

import os
import glob

import scipy

#import tensorflow as tf
import torch
import numpy as np

import matplotlib.pyplot as plt
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

        self._image_size = (260, 346, 1)


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


        self.USE_FRN = True    
        print("USE_FRN =", self.USE_FRN)                         
                                           #####Modell hier wählen
        #if self.USE_FRN:                                  
            #CHECKPOINT_PATH = "/home/carlas/Experiments/event-e2e/CIL/agents/imitation/model/best_model_epoch_8.pth"      #cil/frn unterscheidung
        #else: 
            #CHECKPOINT_PATH = "/home/carlas/Experiments/event-e2e/CIL/agents/imitation/model/best_cil_epoch_9.pth"

        MODEL_DIR = "/home/carlas/Experiments/event-e2e/CIL/agents/imitation/model"
        if self.USE_FRN:
            files = glob.glob(os.path.join(MODEL_DIR, "best_model_epoch_*.pth"))
        else:
            files = glob.glob(os.path.join(MODEL_DIR, "best_cil_epoch_*.pth"))
        CHECKPOINT_PATH = max(files, key=os.path.getmtime) if files else None

        self.model = Mymodel(image_shape=self._image_size, Device=device, use_fusion=self.USE_FRN)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-6) #geändert                                    ###lr anpassen
        self.criterion = torch.nn.MSELoss()
        self.dataset = DDD17Dataset (raw_data[0], raw_data[1], raw_data[2])
        self.dataloader = DataLoader(self.dataset, batch_size=8, shuffle=True, num_workers=0)  #######hier batch size verstellen
        sample = self.dataset[0]
        print("dataset sample len:", len(sample))
        print("dataset sample types:", type(sample), type(sample[0]), type(sample[1]))

        dir_path = os.path.dirname(__file__)

        self._models_path = dir_path + '/model/'
        os.makedirs(self._models_path, exist_ok=True)

        # wird dann in train und test jeweils aufgerufen 
        checkpoint_loss = float("inf")
        checkpoint_epoch = 0
        print("CHECKPOINT_PATH:", CHECKPOINT_PATH)
        print("exists:", os.path.exists(CHECKPOINT_PATH))
        #if os.path.exists(CHECKPOINT_PATH):
            #checkpoint_epoch, checkpoint_loss = self.load_model(CHECKPOINT_PATH)
            #print("Checkpoint loaded")
        #else:
            #print("No checkpoint found — starting fresh")

        #mit checkpoint

        #checkpoint_epoch, checkpoint_loss = self.load_model(CHECKPOINT_PATH)
        #if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
            #print("Loading:", CHECKPOINT_PATH)
            #checkpoint_epoch, checkpoint_loss = self.load_model(CHECKPOINT_PATH)
        #else:
            #print("No checkpoint found — starting fresh")


        #ohne checkpoint
        checkpoint_epoch = 0                                                                        ###### mit/ohne checkpoint
        checkpoint_loss = float("inf")
        print("start ohne checkpoint laden")
        
        self.train(raw_data, device, 5, checkpoint_epoch, checkpoint_loss)                          ######hier Epochennummer ändern
        self.test(device)


    def load_model(self, models_path):

        if not os.path.exists(models_path):
            raise RuntimeError('failed to find the models path')
        
        checkpoint = torch.load(models_path, weights_only=True)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
                  

        return epoch, loss

    #def _compute_action(self, rgb_image, speed_steer):

        #print(torch.max(rgb_image))
        #image_input = rgb_image / 255.0
        #print(torch.max(image_input))

        #predicted_speed, predicted_steer = self.model.forward(input_data=speed_steer,input_image=image_input)

        #control = {}
        #control["steer"] = predicted_steer
        #control["speed"] = predicted_speed

        #return control
    
    def train(self, raw_data, device, num_epochs=1, epoch_start = 0, loss_start = float("inf")):
        self.model.train()
        print(self.model.training)
        self.model.training = True
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
        best_loss = loss_start
        Schwellwert = 0.2
        epoch_losses = []
        first_batch = next(iter(self.dataloader))
        print("batch len", len(first_batch))
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for images, events, targets in self.dataloader:
                images = images.to(device)
                targets = targets.to(device)
                events = events.to(device)

                
                #print(targets.shape)
                #speed_input = targets[:, 0].unsqueeze(1)
                #steer_target = targets[:, 1].unsqueeze(1)
                #output = self.model(images, speed_input, events)
                #print("output", output, "steer_target", steer_target)
                #loss = self.criterion(output, steer_target)
                #print("before:", self.model.fcs[0].weight[0,0].item())
                #print("output:", output[:5].detach().cpu().view(-1))
                #print("target:", steer_target[:5].detach().cpu().view(-1))
                #loss.backward()
                #print("grad norm", self.model.fcs[0].weight.grad.norm().item())
                #self.optimizer.step()
                #self.optimizer.zero_grad()
                #print("after:", self.model.fcs[0].weight[0,0].item())
                self.optimizer.zero_grad()
                speed_input = targets[:, 0].unsqueeze(1)
                steer_target = targets[:, 1].unsqueeze(1)
                output = self.model(images, speed_input, events)

                loss = self.criterion(output, steer_target)
                print("output:", output[:5].detach().cpu().view(-1))
                print("target:", steer_target[:5].detach().cpu().view(-1))
                print("output requires_grad:", output.requires_grad)
                print("loss requires_grad:", loss.requires_grad)
                old = self.model.fcs[0].weight.detach().clone()

                print("before:", self.model.fcs[0].weight[0,0].item())

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                print("grad norm:", self.model.fcs[0].weight.grad.norm().item())

                self.optimizer.step()
                diff = (self.model.fcs[0].weight.detach() - old).abs().sum().item()
                print("weight diff:", diff)
                print("after:", self.model.fcs[0].weight[0,0].item())
                                

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.dataloader)
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, avg_loss: {avg_epoch_loss}")
            if avg_epoch_loss < best_loss:
                torch.save(self.model.state_dict(), self._models_path + "/checkpoint.pth")
                print("Checkpoint saved")
                best_loss = avg_epoch_loss
                if avg_epoch_loss < Schwellwert:
                    if self.USE_FRN:
                        checkpoint_path = os.path.join(self._models_path, f"best_model_epoch_{epoch+1+epoch_start}.pth")#+epochennummer übergeben f str
                    # chechpoint speicher abhänging welche branch
                    else: 
                        checkpoint_path = os.path.join(self._models_path, f"best_cil_epoch_{epoch+1+epoch_start}.pth")
                    torch.save({
                        'epoch': epoch + epoch_start,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_loss,
                    }, checkpoint_path)

                    print("Checkpoint saved:", checkpoint_path)
        print("All epoch losses:", epoch_losses)# als array machen nicht als zahl
        print("best_loss", best_loss)

        epochs = range(1, len(epoch_losses)+1)
        plt.plot(epochs, epoch_losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.grid(True)
        plt.ylim(0, 0.15)
        plt.show()
        

   #def test(self):
        #output = self.model(image, target)
        # criterion / auswertfunktion?
        # mit Ground truth
        #output = self.model(images, events, meta_input)   xxx
    def test(self, device):
        results = []
        self.model.eval()

        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, events, targets in self.dataloader:
                images = images.to(device)
                events = events.to(device)
                targets = targets.to(device)

                speed_input = targets[:, 0].unsqueeze(1)
                steer_target = targets[:, 1].unsqueeze(1)

                output = self.model(images, speed_input, events)

                for gt, pred in zip(steer_target, output):
                    results.append({
                        "ground_truth": gt.item(),
                        "prediction": pred.item(),
                        "error": abs(pred.item() - gt.item())
                    })

                mse = torch.mean((output - steer_target) ** 2)
                mae = torch.mean(torch.abs(output - steer_target))

                total_mse += mse.item()
                total_mae += mae.item()
                num_batches += 1

        avg_mse = total_mse / num_batches
        avg_mae = total_mae / num_batches
        rmse = avg_mse ** 0.5

        print("First 10 predictions:")
        for i, r in enumerate(results[:10]):
            print(
                i,
                "GT:", round(r["ground_truth"], 4),
                "Pred:", round(r["prediction"], 4),
                "Error:", round(r["error"], 4)
            )

        print("Test MSE:", avg_mse)
        print("Test MAE:", avg_mae)
        print("Test RMSE:", rmse)

        self.model.train()
        return avg_mse, avg_mae, rmse, results
    #tatsächliche Vorhersage ausgeben lassen in tabelle bei bestimmten daten
    #dataloader test 3x3 recording durchlaufen lassen
    #quasi dummy daten nochmal testen mit bildern drin, events auf 0, oder andersrum
    pass

