import pickle
import os
from torch.utils.data import Dataset


class NuscRange(Dataset):
  def __init__(self,root):
      self.root = root
      self.file_list = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
      # print('dataset files scanned')

  def __getitem__(self, index):
      file_path = os.path.join(self.root,self.file_list[index])
      with open(file_path, 'rb') as f:
          dict_data = pickle.load(f)
      return dict_data

  def __len__(self):
      return len(self.file_list)