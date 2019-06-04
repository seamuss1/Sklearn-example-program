import os, threading
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

class App:
    def __init__(self):
        for f in ['input','output']:
            if not os.path.isdir(f):
                os.makedirs(f)
        self.models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236}
        print(r'''
 __    __     _                          
/ / /\ \ \___| | ___ ___  _ __ ___   ___ 
\ \/  \/ / _ \ |/ __/ _ \| '_ ` _ \ / _ \
 \  /\  /  __/ | (_| (_) | | | | | |  __/
  \/  \/ \___|_|\___\___/|_| |_| |_|\___|
                                         

''')
    def start(self):
        while True:
            print('''
Enter command:
1) Generate Sample Data
2) Polynomial Regression
3) Ridge Regression
4) Lasso Regression
Q) Quit
''')
            p = input('>>>')
            if p == '1':
                #worker=threading.Thread(target=self.generate_data,daemon=True)
                #worker.start()
                self.generate_data()
            if p == '2':
                self.polynomial_regression()
            if p == '3':
                self.ridge_regression()
            if p == '4':
                self.lasso_regression()
            if p.lower() == 'q':
                print('Goodbye')
                return
    def lasso_regression(self):
        for filename in ['input/'+f for f in os.listdir('input')]:
            data = pd.read_csv(filename)
            for i in range(2,16):
                dataset = data['x']**i
                data['x_{}'.format(i)]=data['x']**i
            predictors=['x']
            predictors.extend(['x_%d'%i for i in range(2,16)])
            alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]
            ind =['rss','intercept'] + ['coef_x_{}'.format(i) for i in range(1,16)]
            coef_matrix_lasso = pd.DataFrame(index=ind)
            models_to_plot = {1e-10:231, 1e-5:232,1e-4:233, 1e-3:234, 1e-2:235, 1:236}
            plt.figure(figsize=(10,7))
            for i in range(10):
                lassoreg = Lasso(alpha=alpha_lasso[i],normalize=True, max_iter=1e5)
                lassoreg.fit(data[predictors],data['y'])
                y_pred = lassoreg.predict(data[predictors])
                plt.ion()
                if alpha_lasso[i] in models_to_plot:
                    plt.subplot(models_to_plot[alpha_lasso[i]])
                    plt.tight_layout()
                    plt.plot(data['x'],y_pred)
                    plt.plot(data['x'],data['y'],'.')
                    plt.title('Plot for alpha: {}'.format(alpha_lasso[i]))
                plt.draw()
                plt.pause(0.1)
                rss = sum((y_pred-data['y'])**2)
                interc = lassoreg.intercept_
                coef = lassoreg.coef_
                dataline = [rss, interc]+[f for f in coef]
                [dataline.append(None) for f in range(len(ind)-len(dataline))]
                coef_matrix_lasso['alpha_{}'.format(alpha_lasso[i])] = dataline
            coef_matrix_lasso.to_csv('output/lasso_coefficients.csv')
            print(coef_matrix_lasso.head())
            plt.savefig('output/lasso_regression_figure.png')
            plt.pause(2)
            plt.close()
    def ridge_regression(self):
        for filename in ['input/'+f for f in os.listdir('input')]:
            data = pd.read_csv(filename)
            predictors=['x']
            predictors.extend(['x_%d'%i for i in range(2,16)])
            models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
            for i in range(2,16):
                dataset = data['x']**i
                data['x_{}'.format(i)]=data['x']**i
            alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
            ind=['rss','intercept'] + ['coef_x_{}'.format(i) for i in range(1,16)]
            coef_matrix_ridge = pd.DataFrame(index=ind)
            plt.figure(figsize=(10,7))
            for i in range(10):
                ridgereg = Ridge(alpha=alpha_ridge[i],normalize=True)
                ridgereg.fit(data[predictors],data['y'])
                y_pred = ridgereg.predict(data[predictors])
                plt.ion()
                
                if alpha_ridge[i] in models_to_plot:
                    plt.subplot(models_to_plot[alpha_ridge[i]])
                    plt.tight_layout()
                    plt.plot(data['x'],y_pred)
                    plt.plot(data['x'],data['y'],'.')
                    plt.title('Plot for alpha: {}'.format(alpha_ridge[i]))
                    plt.pause(0.1)
                rss = sum((y_pred-data['y'])**2)
                interc = ridgereg.intercept_
                coef = ridgereg.coef_
                dataline = [rss, interc]+[f for f in coef]
                [dataline.append(None) for f in range(len(ind)-len(dataline))]
                coef_matrix_ridge['alpha+{}'.format(alpha_ridge[i])] = dataline
            print(coef_matrix_ridge.head())
            coef_matrix_ridge.to_csv('output/ridge_coefficients.csv')
            plt.savefig('output/ridge_regression_figure.png')
            plt.pause(2)
            plt.close()
    def polynomial_regression(self):
        for filename in ['input/'+f for f in os.listdir('input')]:
            data = pd.read_csv(filename)
            for i in range(2,16):
                dataset = data['x']**i
                data['x_{}'.format(i)]=data['x']**i
            coef_matrix_simple = pd.DataFrame(index=['rss','intercept']+['coef'+str(f) for f in range(1,16)])
            plt.figure(figsize=(10,7))
            for line in data:
                
                if not line.startswith('x'):
                    continue
                #print(line)
                try:
                    power=int(line[2:])
                except ValueError:
                    power=1
                predictors=['x']
                if power>=2:
                    predictors.extend(['x_{}'.format(i) for i in range(2,power+1)])
                x = data[predictors]
                y = data['y']
                linreg = LinearRegression(normalize=True)
                linreg.fit(x,y)
                y_pred = linreg.predict(x)
                rss = sum((y_pred-data['y'])**2)
                interc = linreg.intercept_
                coef = linreg.coef_
                dataline = [rss, interc]+[f for f in coef]
                [dataline.append(None) for f in range(17-len(dataline))]
                coef_matrix_simple[line] = dataline
                plt.ion()
                if power in self.models_to_plot:
                    plt.subplot(self.models_to_plot[power])
                    plt.tight_layout()
                    plt.plot(data['x'],y_pred)
                    plt.plot(data['x'],data['y'],'.')
                    plt.title('Plot for power: {}'.format(power))

                plt.draw()
                plt.pause(0.1)
        coef_matrix_simple.to_csv('output/coefficients.csv')
        print(coef_matrix_simple.head())
        data.to_csv('output/data.csv')
        plt.savefig('output/basic_regression_figure.png')
        plt.pause(2)
        plt.close()
    def generate_data(self):
        x = np.array([i*np.pi/180 for i in range(60,300,4)])
        y = np.sin(x) + np.random.normal(0,0.15,len(x))
        filename = 'input/Randomized Sine Data.csv'
        with open(filename,'w') as file:
            file.write('x,y\n')
            for x1,y1 in zip(x,y):
                line = '{},{}\n'.format(x1,y1)
                #print(line)
                file.write(line)
        self.data = pd.read_csv(filename)
        plt.ion()
        plt.plot(x,y)
        plt.title('Generated Sine function with Random Variation')
        plt.draw()
        plt.pause(0.01)
        
if __name__ == '__main__':
    app = App()
    app.start()
