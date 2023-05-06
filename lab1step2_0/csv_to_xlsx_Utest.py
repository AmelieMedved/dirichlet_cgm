import pandas as pd
import os.path
import xlsxwriter
outData = pd.read_csv("tableUtest.csv", sep = ',', index_col=0, encoding='cp1251')
if(os.path.exists('tableUtest.xlsx')):
  os.remove('tableUtest.xlsx')
#print(outData) # Посмотреть на таблицу pandas.
#outData.drop(outData.columns[outData.iloc[0].size-1], axis = 1, inplace = True) # Эта строка удаляет последний столбец, если данные в .csv в конце каждой строки имеют запятую.
writer = pd.ExcelWriter('tableUtest.xlsx',engine='xlsxwriter') 
outData.to_excel(writer,'tableUtest') 
writer.save()