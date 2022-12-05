#Modul Library
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# #modul library data testing dan training
from sklearn.model_selection import train_test_split

# #modul library score tingkat akurasi
from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/SitiUlunNuhaKarimah127/datamining/main/predic_tabel.csv"
data = pd.read_csv(url)
y = data['Hasil'] 

st.write("""
         Nama   : Siti Ulun Nuha Karimah\n
         NIM    : 200411100127\n
         Kelas  : Data Mining A\n
         """)
st.title('Sistem Pendeteksi Penyakit Paru-Paru')
deskripsi, dataset, preprocessing, modeling, implementation= st.tabs(["Info", "Dataset", "Preprocessing", "Modeling", "Implementation"])

with deskripsi:
    st.write("""
             Penyakit paru-paru adalah kondisi yang membuat paru-paru tidak dapat berfungsi secara normal.
             Penyakit paru-paru merupakan gangguan kesehatan paling umum di dunia, yang menyerang pasien dari segala usia.
             Sebagian besar penyakit paru-paru disebabkan oleh merokok. Asap rokok mengandung racun yang menganggu kinerja tubuh dalam menyaring udara yang masuk dan keluar dari paru-paru.
             """)
    st.write("""
             Jenis - jenis penyakit paru-paru yang paling umum:
             1. Penyakit Paru Obstruktif Kronis (PPOK)
             2. Pneumonia
             3. Tuberkulosis
             4. Kanker Paru-Paru 
             """)
    st.write("""
             Penderita Penyakit Paru-paru bisa mengalami gejala berupa:
             
             1. Sulit Bernapas
             2. Batuk Kronis
             3. Nyeri Dada
             """)

with dataset:
    st.write("""Link Dataset : https://www.kaggle.com/datasets/andot03bsrc/dataset-predic-terkena-penyakit-paruparu""")
    ("""
     Aktivitas merokok merupakan salah satu penyebab sumber penyakit, tidak hanya berdampak aktif
perokok tetapi orang-orang di sekitar perokok atau perokok pasif juga terpengaruh. Pasif
perokok lebih mungkin mengalami efek penyakit seperti perokok aktif.
Namun jika 1% dari populasi manusia yang ada menjadi perokok pasif maka
sejumlah dokter spesialis paru yang ada tidak akan mampu menanganinya. Ini adalah masalah yang
harus ditangani. Pengguna juga dapat melakukan diagnosa awal terhadap gejala yang diderita
sebagai perlakuan mereka melalui Sistem Pakar. Dalam penelitian ini, sistem pakar menggunakan metode faktor kepastian yang dapat
memberikan kepastian suatu fakta. Perhitungan dilakukan berdasarkan nilai keyakinan seorang ahli terhadap gejala a
penyakit. Sistem pakar yang dihasilkan diberi nama Diagperosif dimana sistem mendiagnosa penyakit berdasarkan gejalanya
dimasukkan oleh pengguna. Penyakit yang dapat didiagnosis dengan Diagperosif adalah asma, bronkitis, polisit, dan kanker paru-paru.
     """)
    
    st.write("""
             Keterangan Kolom :
             1. Usia: Di usia terdapat 2 pilihan yaitu muda dan tua
             2. Jenis_Kelamin: Jenis kelamin memiliki 2 pilihan yaitu pria dan wanita
             3. Merokok: Di dalam kolom merokok ada 2 pilihan yaitu aktif dan pasif
             4. Bekerja: Kolom bekerja terdapat 2 pilihan yaitu Ya atau tidak
             5. Rumah_Tangga: Pada kolom Rumah_Tangga terdapat 2 pilihan Ya atau tidak
             6. Aktivitas_Begadang: Kolom Aktivitas_Begadang memiliki 2 pilihan yaitu Ya atau tidak
             7. Aktivitas_Olahraga: Di dalam kolom Aktivatas Olahraga ada 2 pilihan yaitu Jarang dan Sering
             8. Asuransi: Pada kolom Asuransi terdapat 2 pilihan yaitu ada dan tidak
             9. Penyakit_Bawaan: Di kolom Penyakit_Bawaan terdapat 2 pilihan yaitu ada dan tidak  
             """)
    st.write("""
             Di dalam dataset terdapat 3000 data dan 11  kolom
             """)
    df = pd.DataFrame(data)
    df


with preprocessing:
    st.write("""
             Data Sebelum Kolom Nomor dan Hasil Dihilangkan
             """)
    df = pd.DataFrame(data)
    df
    
    H = df.drop(columns=['No'])
    X = H.drop(columns=['Hasil'])
    st.write("""
             Data Sesudah Kolom Nomor dan Hasil Dihilangkan
             """)
    X
    st.write("""
             Data setelah dilakukan preprocessing menggunakan tahap normalisasi data string ke kategori
             """)
    #  Tahap Normalisasi data string ke kategori
    X = pd.DataFrame(X)
    X['Usia'] = X['Usia'].astype('category')
    X['Jenis_Kelamin'] = X['Jenis_Kelamin'].astype('category')
    X['Merokok'] = X['Merokok'].astype('category')
    X['Bekerja'] = X['Bekerja'].astype('category')
    X['Rumah_Tangga'] = X['Rumah_Tangga'].astype('category')
    X['Aktivitas_Begadang'] = X['Aktivitas_Begadang'].astype('category')
    X['Aktivitas_Olahraga'] = X['Aktivitas_Olahraga'].astype('category')
    X['Asuransi'] = X['Asuransi'].astype('category')
    X['Penyakit_Bawaan'] = X['Penyakit_Bawaan'].astype('category')
    cat_columns = X.select_dtypes(['category']).columns
    X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
    X
with modeling:
    knn, naive_bayes, decission_tree= st.tabs(["KNN", "Naive_Bayes", "Decission_Tree"])
    #Encoder Label (Mengubah label ke kategori)
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y_baru = le.fit_transform(y)
    
    #Split Dataset
    X_train, X_test, y_train, y_test=train_test_split(X, y_baru, test_size=0.2, random_state=1)
    with knn:
        #Inisialisasi Model KNN
        my_param_grid = {'n_neighbors':[2,3,5,7], 'weights': ['distance', 'uniform']}
        GridSearchCV(estimator=KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
        knn = GridSearchCV(KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
        knn.fit(X_train, y_train)
        
        pred_test = knn.predict(X_test)
        vknn = f'acuracy = {accuracy_score(y_test, pred_test) * 100 :.2f}'
        vknn
        
        filenameModelKnnNorm = 'modelKnnNorm.pkl'
        joblib.dump(knn, filenameModelKnnNorm)
    
    with naive_bayes:    
        #Inisialisasi Model Gaussian
        # training the model on training set
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        filenameModelGau = 'modelGau.pkl'
        joblib.dump(gnb, filenameModelGau)
        # making predictions on the testing set gausian
        g_pred = gnb.predict(X_test)
        vg = f'acuracy = {accuracy_score(y_test, pred_test) * 100 :.2f}'
        vg
    
    with decission_tree:
        #Inisialisasi Model Decission Tree
        gnb = DecisionTreeClassifier()
        gnb.fit(X_train, y_train)
        filenameModelDec = 'modelDec.pkl'
        joblib.dump(gnb, filenameModelDec)
        #making predictions on the testing set decissiontree
        d_pred = gnb.predict(X_test)
        vd = f'acuracy = {accuracy_score(y_test, pred_test) * 100 :.2f}'
        vd
    
with implementation:
    usia = st.selectbox(
    'Usia',
    ('Muda', 'Tua'))
    
    jenis_kelamin = st.selectbox(
    'Jenis Kelamin',
    ('Pria', 'Wanita'))
    
    merokok = st.selectbox(
    'Merokok',
    ('Aktif', 'Pasif'))
    
    bekerja = st.selectbox(
    'Bekerja',
    ('Tidak', 'Ya'))
    
    rumah_tangga = st.selectbox(
    'Rumah Tangga',
    ('Tidak', 'Ya'))
    
    aktivitas_begadang = st.selectbox(
    'Aktivitas Begadang',
    ('Tidak', 'Ya'))
    
    aktivitas_olahraga = st.selectbox(
    'Aktivitas Olahraga',
    ('Jarang', 'Sering'))
    
    asuransi = st.selectbox(
    'Asuransi',
    ('Ada', 'Tidak'))
    
    penyakit_bawaan = st.selectbox(
    'Penyakit Bawaan',
    ('Ada', 'Tidak'))
    
    ind_usia = ('Muda','Tua').index(usia)
    ind_jenis_kelamin = ('Pria', 'Wanita').index(jenis_kelamin)
    ind_merokok = ('Aktif', 'Pasif').index(merokok)
    ind_bekerja = ('Tidak', 'Ya').index(bekerja)
    ind_rumah_tangga = ('Tidak', 'Ya').index(rumah_tangga)
    ind_aktivitas_begadang = ('Tidak', 'Ya').index(aktivitas_begadang)
    ind_aktivitas_olahraga = ('Jarang', 'Sering').index(aktivitas_olahraga)
    ind_asuransi = ('Ada', 'Tidak').index(asuransi)
    ind_penyakit_bawaan =('Ada', 'Tidak').index(penyakit_bawaan)
    
    
    #Predict Input (Memuat Data Baru)

    a = np.array([[ind_usia, ind_jenis_kelamin, ind_merokok, ind_bekerja, ind_rumah_tangga, ind_aktivitas_begadang, ind_aktivitas_olahraga, ind_asuransi, ind_penyakit_bawaan ]])
    # test_data = np.array(a).reshape(1, -1)
    # test_data
    data_inputan = pd.DataFrame(a, columns =['Usia', ' Jenis_Kelamin', 'Merokok', 'Bekerja', 'Rumah_Tangga', 'Aktivitas_Begadang', 'Aktivitas_Olahraga',  'Asuransi', 'Penyakit_bawaan'])    
    label = {0:'tidak terdeteksi penyakit paru-paru', 1:'terdeteksi penyakit paru-paru'}
    KNN, Naive_Bayes, Decission_tree= st.tabs(["Knn", "Naive_Bayes", "Decission_Tree"])
    
with KNN:
    #Load Model KNN
    knn = joblib.load(filenameModelKnnNorm)
    inp_pred = knn.predict(data_inputan)
    label[inp_pred[0]]
 
with Naive_Bayes:    
    #Load Model Gaussian
    gnb = joblib.load(filenameModelGau)
    inp_pred = gnb.predict(data_inputan)
    label[inp_pred[0]]

with Decission_tree:    
    #Load Model DecissionTree
    dnb = joblib.load(filenameModelDec)
    inp_pred = dnb.predict(data_inputan)
    label[inp_pred[0]]

    


        
