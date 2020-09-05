from demo.liner_regression import linear_regression
from demo.keras.linear_regression import linear_regression as linear_regression_keras
from misc import measure_time


def main():
  params = {'a': 50, 'b': -30}
  print(f'our implementation time: {measure_time(linear_regression, **params)}')  # 0.109s
  print(f'keras(only CPU) implementation time: {measure_time(linear_regression_keras, **params)}')  # 0.716s


if __name__ == '__main__':
  main()

