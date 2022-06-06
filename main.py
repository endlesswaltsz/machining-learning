from ast import main
import csv
from lib2to3.pytree import LeafPattern
import numpy as np
import configparser
config = configparser.ConfigParser()





def write_config(learning_rate, house_age_rate, distance_to_nearest_market_rate, number_of_convenience_store,const_rate):
    config['DEFUALT'] = {
        'learning_rate' : learning_rate,
        'house_age_rate': house_age_rate,
        'distance_to_nearest_market_rate':distance_to_nearest_market_rate,
        'number_of_convenience_store':number_of_convenience_store,
        'const_rate':const_rate,
        'step':0.00000001,
    }
    with open('config.ini', 'w') as configfile:
        config.write(configfile)


def load_config():
    config.read('config.ini')
    return config
    # learning_rate = config['DEFUALT']['learning_rate']
    # house_age_rate = config['DEFUALT']['house_age_rate']
    # distance_to_nearest_market_rate = config['DEFUALT']['distance_to_nearest_market_rate']
    # number_of_convenience_store = config['DEFUALT']['number_of_convenience_store']
    # step = config['DEFUALT']['step']

# def get_preset_ratio():
#   a = []
#   b = []
#   for i in range(0,3):
#     a.append([float(data[i]['X2 house age']),float(data[i]['X3 distance to the nearest MRT station']),float(data[i]['X4 number of convenience stores'])])
#     b.append([float(data[i]['Y house price of unit area'])])
#   print(np.array(a))
#   print(np.array(b))
#   print(np.linalg.solve(np.array(a), np.array(b)))

def get_data()-> list:
  data=[]
  with open('Real estate.csv', newline='') as csvfile:
    dic=csv.DictReader(csvfile)
    for row in dic:
      data.append(row)
  return data


def gradient(x1, x2, x3, y, θ0, θ1, θ2, θ3, param):
  return (θ0 + θ1*x1 + θ2*x2 + θ3*x3 -y) * param

def quadradic_deviation(x1, x2, x3, y, θ0, θ1, θ2, θ3):
  return (θ0 + θ1*x1 + θ2*x2 + θ3*x3 -y)**2




def main():
  '''
  model: house_price_of_unit_area = θ0 + Θ1 * house_age + θ2 * distance_to_nearest_market + θ3 * number_of_convenience_store
  preset_ratio:
    const = 0
    house_age = 0.552
    distance_to_nearest_market=0.057
    number_of_convenience_store=1.537
  '''
  config = load_config()
  learning_rate = float(config['DEFUALT']['learning_rate'])
  const_rate = float(config['DEFUALT']['const_rate'])
  house_age_rate = float(config['DEFUALT']['house_age_rate'])
  distance_to_nearest_market_rate=float(config['DEFUALT']['distance_to_nearest_market_rate'])
  number_of_convenience_store=float(config['DEFUALT']['number_of_convenience_store'])
  step=float(config['DEFUALT']['step'])
  data = get_data()
  loss=None
  try:
    while True:
      loss_result = sum([quadradic_deviation(float(i['X2 house age']),
                    float(i['X3 distance to the nearest MRT station']),
                    float(i['X4 number of convenience stores']),
                      float(i['Y house price of unit area']),
                      const_rate,
                      house_age_rate,
                      distance_to_nearest_market_rate,
                      number_of_convenience_store) for i in data])
      print(f'Variance:{loss_result}')
      if not loss:
        loss = loss_result
      else:
        if loss_result > loss:
          # print(const_rate, house_age_rate, distance_to_nearest_market_rate, number_of_convenience_store)
          # return
          learning_rate -= step
          write_config(learning_rate, house_age_rate, distance_to_nearest_market_rate, number_of_convenience_store, const_rate)
        else:
          #update params
          const_rate = const_rate - learning_rate * (1/400) * sum(
            [gradient(float(i['X2 house age']),
                    float(i['X3 distance to the nearest MRT station']),
                    float(i['X4 number of convenience stores']),
                      float(i['Y house price of unit area']),
                      const_rate,
                      house_age_rate,
                      distance_to_nearest_market_rate,
                      number_of_convenience_store, 1) for i in data]
                      )
          # print(const_rate)
          house_age_rate = house_age_rate - learning_rate * (1/400) * sum(
              [gradient(float(i['X2 house age']),
                      float(i['X3 distance to the nearest MRT station']),
                      float(i['X4 number of convenience stores']),
                        float(i['Y house price of unit area']),
                        const_rate,
                        house_age_rate,
                        distance_to_nearest_market_rate,
                        number_of_convenience_store, float(i['X2 house age'])) for i in data]
                        )
          # print(house_age_rate)
          distance_to_nearest_market_rate = distance_to_nearest_market_rate - learning_rate * (1/400) * sum(
              [gradient(float(i['X2 house age']),
                      float(i['X3 distance to the nearest MRT station']),
                      float(i['X4 number of convenience stores']),
                        float(i['Y house price of unit area']),
                        const_rate,
                        house_age_rate,
                        distance_to_nearest_market_rate,
                        number_of_convenience_store, float(i['X3 distance to the nearest MRT station'])) for i in data]
                        )
          number_of_convenience_store = number_of_convenience_store - learning_rate * (1/400) * sum(
              [gradient(float(i['X2 house age']),
                      float(i['X3 distance to the nearest MRT station']),
                      float(i['X4 number of convenience stores']),
                        float(i['Y house price of unit area']),
                        const_rate,
                        house_age_rate,
                        distance_to_nearest_market_rate,
                        number_of_convenience_store, float(i['X4 number of convenience stores'])) for i in data]
                        )
  except KeyboardInterrupt:      
    write_config(learning_rate, house_age_rate, distance_to_nearest_market_rate, number_of_convenience_store, const_rate)
  
    



if __name__ == "__main__":
  # for i in range(0,2):
  main()

    