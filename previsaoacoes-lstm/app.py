
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import pandas_datareader as web
import datetime as dt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#load data (carregando dados)
company = 'FB'

start = dt.datetime(2012,1,1)
end = dt.datetime(2022,8,24)

data = web.DataReader(company, 'yahoo',start,end)

# Prepare data (preparando dados)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data =  scaler.fit_transform(data['Close'].values.reshape(-1,1))

predictions_days = 60  #Quantidade de amostras

x_train = []
y_train = []

for x in range(predictions_days, len(scaled_data)):
    x_train.append(scaled_data[x-predictions_days:x,0])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))

#buit the model (construindo o modelo
model = Sequential()

model.add(LSTM(units=50,return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #predction of the next close price

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

""" testing the model accuracy on existing data"""

#load test data (carregando dados de teste)
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo',test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - predictions_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#make predictions on test dada (Fazendo predições nos dados de teste)

x_test = []

for x in range(predictions_days, len(model_inputs)):
    x_test.append(model_inputs[x-predictions_days:x,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

predictions_prices = model.predict(x_test)
predictions_prices = scaler.inverse_transform(predictions_prices)



#predict the next day (Fazendo predição do dia seguinte)
real_data = [model_inputs[len(model_inputs) + 1 - predictions_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

print(f"Prediction {prediction}")




#plot the test predictions (Plotando as predições de teste no grafico)
def creat_plot(actual_prices,company):
    plt.plot(actual_prices, color='red', label=f"Preço {company} Ação")
    plt.plot(predictions_prices, color='green', label= f"Preço {company} Predição")
    plt.title(f"{company} Preço")
    plt.xlabel('Time')
    plt.ylabel(f"{company} Preço")
    plt.legend()
    return plt.gcf()



#layout pysimplegui

layout = [ 
    [sg.Text('Grafico de linhas')],
    [sg.Canvas(size=(500,500), key ="-CANVAS-")],
    [sg.Exit(size=(20,2),button_color=('grey')),
    sg.Button('Previsão ',size=(20,2),button_color=('grey'), key='prev')]

]
def draw_figure(canvas,figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure,canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='bottom',fill='both', expand=1)
    return figure_canvas_agg

window = sg.Window('Grafico de linhas', layout, finalize=True, element_justification='center')

draw_figure(window["-CANVAS-"].TKCanvas, creat_plot(actual_prices,company))


while True:
    event,values = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == 'prev':
        sg.popup(f'A previsão para amanhã será: {prediction}')
window.close()







