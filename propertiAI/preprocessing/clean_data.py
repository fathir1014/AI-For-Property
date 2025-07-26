from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data_classifier(data) :

  location = LabelEncoder()
  data['Lokasi'] = location.fit_transform(data['Lokasi'])

  credit_history = LabelEncoder()
  data['RiwayatKredit'] = credit_history.fit_transform(data['RiwayatKredit'])

  numeric_columns = ['LuasBangunan', 'JumlahKamar', 'TahunBangun', 'DayaListrik', 'Penghasilan', 'HargaProperti' ]
  scaler = StandardScaler()
  data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

  X = data.drop(columns=['LayakKredit'])
  y = data['LayakKredit']

  return X,y

def clean_data_regression(data) :
  data = data.drop(columns=['LayakKredit'])

  credit_history = LabelEncoder()
  data['RiwayatKredit'] = credit_history.fit_transform(data['RiwayatKredit'])

  numeric_columns = ['LuasBangunan', 'JumlahKamar', 'TahunBangun', 'DayaListrik', 'Penghasilan']
  scaler = StandardScaler()
  data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

  X = data.drop(columns=['HargaProperti'])
  y = data['HargaProperti']

  return X, y

