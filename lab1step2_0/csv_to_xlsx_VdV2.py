import pandas as pd
import os.path
outData = pd.read_csv("tableVdV2.csv", sep = ',', index_col=0, encoding='cp1251')
if(os.path.exists('tableVdV2.xlsx')):
  os.remove('tableVdV2.xlsx')
#print(outData) # Посмотреть на таблицу pandas.
#outData.drop(outData.columns[outData.iloc[0].size-1], axis = 1, inplace = True) # Эта строка удаляет последний столбец, если данные в .csv в конце каждой строки имеют запятую.
writer = pd.ExcelWriter('tableVdV2.xlsx',engine='xlsxwriter') 
outData.to_excel(writer,'tableVdV2') 
writer.save()