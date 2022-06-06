import csv
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

def get_data()-> list:
  data=[]
  with open('t1.csv', newline='') as csvfile:
    dic=csv.DictReader(csvfile)
    for row in dic:
      data.append(row)
  return data


learning_rate = float(config['DEFUALT']['learning_rate'])
const_rate = float(config['DEFUALT']['const_rate'])
house_age_rate = float(config['DEFUALT']['house_age_rate'])
distance_to_nearest_market_rate=float(config['DEFUALT']['distance_to_nearest_market_rate'])
number_of_convenience_store=float(config['DEFUALT']['number_of_convenience_store'])

data = get_data()



def main():
    l=[]
    for row in data:
        predict = const_rate + house_age_rate * float(row['X2 house age'])
        + distance_to_nearest_market_rate * float(row['X3 distance to the nearest MRT station'])
        + number_of_convenience_store * float(row['X4 number of convenience stores'])
        a = float(row['Y house price of unit area'])
        l.append((predict-a)**2)
        print(f'预测:{round(predict,2)}, 实际：{a}')
    res = sum(l)
    print(f'方差：{res}')

main()
