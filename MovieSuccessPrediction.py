from tkinter import *
from tkinter import messagebox
from tkinter import filedialog as FileDialog

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
class Aplicacion:
    window = Tk()
    window.title("Movie Success Prediction") 
    window.geometry('700x500')
    #Director
    lblD = Label(window, text="Nombre del Director: ")
    lblD.grid(column = 0, row = 0)
    txtD = Entry(window, width = 20)
    txtD.grid(column = 1, row = 0)
    #Actor
    lblA = Label(window, text="Nombre del actor: ")
    lblA.grid(column = 0, row = 2)
    txtA = Entry(window, width = 20)
    txtA.grid(column = 1, row = 2)
    #Genero
    lblG = Label(window, text="Genero: ")
    lblG.grid(column = 0, row = 4)
    lblGE = Label(window, text="(Action|Thriller|Adventure)")
    lblGE.config(fg = "grey")
    lblGE.grid(column = 6, row = 4)
    txtG = Entry(window, width = 20)
    txtG.grid(column = 1, row = 4)
    #IMDB Rating
    lblR = Label(window, text="IMDB Rating: ")
    lblR.grid(column = 0, row = 6)
    lblRE = Label(window, text="(1 - 10(Decimales))")
    lblRE.config(fg = "grey")
    lblRE.grid(column = 6, row = 6)
    txtR = Entry(window, width = 20)
    txtR.grid(column = 1, row = 6)
    #Presupuesto
    lblP = Label(window, text = "Presupuesto($dlls): ")
    lblP.grid(column = 0, row = 8)
    txtP = Entry(window, width = 20)
    txtP.grid(column = 1, row = 8)
    #gross
    lblRet = Label(window, text = "Retribución($dlls): ")
    lblRet.grid(column = 0, row = 10)
    txtRet = Entry(window, width = 20)
    txtRet.grid(column = 1, row = 10)
    #porcentaje de ganancia
    lblPG = Label(window, text="Porcentaje de ganancia: ")
    lblPG.grid(column = 0, row = 12)
    lblPGE = Label(window, text="%")
    lblPGE.config(fg = "grey")
    lblPGE.grid(column = 2, row = 12)
    lblPG2 = Label(window, text = "")
    lblPG2.grid(column = 1, row = 12)
    #Resultado
    lblRes = Label(window, text = "Predicción:   ")
    lblRes.grid(column = 0, row = 16)
    #Resultado
    lblRes = Label(window, text = "Predicción:   ")
    lblRes.grid(column = 0, row = 16)
    #Porcentaje
    lblPres = Label(window, text = "Presición del algoritmo:   ")
    lblPres.grid(column = 0, row = 18)
    lblRes1 = Label(window, text = "")
    lblRes1.grid(column = 1, row = 16)
    lblPress1 = Label(window, text = "")
    lblPress1.grid(column = 1, row = 18)
    file = "";
    def __init__(self):
        self.btn = Button(self.window, text="Predecir!",command=self.predecir)
        self.btn.grid(column=0, row = 14)
        self.btn = Button(self.window, text="Abrir!",command=self.abrir)
        self.btn.grid(column=0, row = 40)
        self.window.mainloop();

    def abrir(self):
        self.file = FileDialog.askopenfilename(
        initialdir="C:", 
        filetypes=(
            ("Ficheros CSV", "*.csv"),
            ("Todos los ficheros","*.*")
        ), 
        title = "Abrir un fichero") 
        
    def predecir(self):
        try:
            ds = pd.read_csv(self.file)
            try:
                #Fividir los datos en X (variable independiente) y Y (Variable dependiente)
                x = ds.iloc[:, :-1].values
                y = ds.iloc[:, 11].values

                #codificacion de datos y de la variable dependiente
                number = LabelEncoder()
                nameencoder=LabelEncoder()
                actor1encoder=LabelEncoder()
                actor2encoder=LabelEncoder()
                actor3encoder=LabelEncoder()
                genresencoder=LabelEncoder()
                imdbscoreencoder=LabelEncoder()
                budgetencoder=LabelEncoder()
                grossencoder=LabelEncoder()
                profitencoder =LabelEncoder()

                #Codificación de cada categoria
                ds['director_name'] = nameencoder.fit_transform(ds['director_name'])
                ds['actor_1_name'] = actor1encoder.fit_transform(ds['actor_1_name'])
                ds['actor_2_name'] = actor2encoder.fit_transform(ds['actor_2_name'].astype(str))
                ds['actor_3_name'] = actor3encoder.fit_transform(ds['actor_3_name'].astype(str))
                ds['genres'] = genresencoder.fit_transform(ds['genres'])
                """ds['imdb_score'] = imdbscoreencoder.fit_transform(ds['imdb_score'])
                ds['budget'] = budgetencoder.fit_transform(ds['budget'])
                ds['gross'] = grossencoder.fit_transform(ds['gross'])
                ds['profit_percent'] = profitencoder.fit_transform(ds['profit_percent'])"""

                features = ["director_name", "actor_1_name", "genres","imdb_score","budget","gross","profit_percent"]

                #Codificacion de la variable dependiente
                labelencoder_y = LabelEncoder()
                y = labelencoder_y.fit_transform(y)

                #Separando los datos en el conjunto de entrenamiento y de pruebas
                #test_size = 0.2 esto es el 20%
                from sklearn.model_selection import train_test_split
                x_train, x_test, y_train, y_test = train_test_split(ds[features], y, test_size = 0.2, random_state = 0)

                #Escala de datos
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)

                #Crear el clasificador Gaussian
                model = GaussianNB()

                #Entrenando el modelo usando los conjuntos entrenados
                model.fit(x_train, y_train)

                #Predecir los valores
                nb_predict_train = model.predict(x_test)

                #Importar las librerias del rendimiento de las metricas
                from sklearn import metrics
                #predict = ["James Cameron","CCH Pounder","Action|Adventure|Fantasy|Sci-Fi",7.9,237000000,760505847,2.20888543]

                #Guardar lo que se escriba en las cajas de texto
                #Predecir la salida 
                #Durante el entrenamiento se dan algunas caracteristicas
            
                self.director_name = self.txtD.get()
                self.actor_name = self.txtA.get()
                self.genre = self.txtG.get()
                self.imdb_rating = float(self.txtR.get())
                self.budget = float(self.txtP.get())
                self.gross = float(self.txtRet.get())
                self.profit_percent = (float(self.gross/self.budget) * 100) /100
                self.lblPG2['text'] = (self.profit_percent) * 100
            
                predict=[self.director_name,self.actor_name,self.genre,self.imdb_rating,self.budget,self.gross,self.profit_percent]
                print(predict)

                predict[0] = nameencoder.transform([predict[0]])
                predict[1] = actor1encoder.transform([predict[1]])
                predict[2] = genresencoder.transform([predict[2]])
                """predict[3] = imdbscoreencoder.transform([predict[3]])
                predict[4] = budgetencoder.transform([predict[4]])
                predict[5] = grossencoder.transform([predict[5]])
                predict[6] = profit_percent.transform([predict[6]])"""

                predict = scaler.transform([predict])
                prediction = model.predict(predict)
                
                if prediction == 1:
                    print("HIT")
                else:
                    print("Flop")

                    #Presición del algoritmo
                    #print("Presicion: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_train)))
                if prediction == 1:
                    self.lblRes1['text'] = "Éxito"
                else:
                    self.lblRes1['text'] = "Fracaso"
                res = (float("{0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_train)))*100,"%")    
                self.lblPress1.config(text = res)
            except:
                messagebox.showinfo(title="Advertencia",message="Algun campo esta vacio o algun dato esta mal")
        except:
            messagebox.showinfo(title="Advertencia",message="Aun no hay ningun archivo csv")
        

          
if __name__ == "__main__":    
    app = Aplicacion()