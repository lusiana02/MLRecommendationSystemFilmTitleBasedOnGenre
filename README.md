# _MLRecommendationSystemFilmTitleBasedOnGenre_
_Project Machine Learning_ Terapan _Recommendation System Film Title Based On Genre_

# _Project Overview_
Pada proyek ini akan membahas tentang sebuah perusahaan yang bergerak di industri perfilman yang ingin meningkatkan _traffic platform film streaming_ mereka, oleh karena itu perusahaan akan mencoba membuat sistem rekomendasi dengan menerapkan pendekatan _Machine Learning_ untuk merekomendasi film-film yang mereka sediakan berdasarkan genre film untuk para _customer_ mereka.

Tujuan dirancangnya sebuah sistem rekomendasi adalah untuk memberikan rekomendasi yang sesuai dengan kebutuhan pengguna. Sistem rekomendasi adalah fitur-fitur dan teknik-teknik pada perangkat lunak yang menyediakan sesuatu hal yang berguna untuk _user_. Sistem rekomendasi juga menyediakan rekomendasi-rekomendasi dari beberapa item yang berpotensi menarik untuk pengguna[1]. Hal ini menjadi penting mengingat tidak semua pengguna dapat menentukan pilihannya saat pertama kali menggunakan suatu aplikasi atau layanan.

## Latar Belakang
Film merupakan salah satu bentuk media komunikasi massa dari berbagai macam teknologi dan berbagai unsur-unsur kesenian. Perpaduan yang seimbang dan harmonis antara seni sastra, seni musik, seni peran dan komedi dikemas menjadi satu dalam bentuk film[2]. Menonton film adalah alternatif hiburan yang sering dipilih ketika merasa penat atau bosan dengan rutinitas. Apalagi kini banyak tersedia platform digital dari yang gratis hingga berbayar yang menawarkan beragam genre film dan bisa ditonton di mana saja melalui ponsel.

Salah satu faktor yang mempengaruhi seseorang untuk menonton film adalah melihat dari kategori genrenya. Contohnya orang yang menyukai film Avengers kemungkinan juga akan menyukai film Thor, karena kedua film tersebut memiliki genre yang sama yaitu _Action_. Oleh karena itu di proyek kali ini perusahaan akan membuat sebuah sistem rekomendasi film berdasarkan genre menggunakan pendekatan _Machine Learning_ guna meningkatkan _traffic platform_ mereka.

# _Bussiness Understanding_

## _Problem Statement_
- Model _Machine Learning_ apa yang cocok untuk menyelesaikan permasalahan rekomendasi film berdasarkan genre?
- Bagaimana cara menentukan hasil rekomendasi suatu model _Machine Learning_ dapat dikatakan baik?

## _Goals_
- Model yang cocok untuk menyelesaikan masalah rekomendasi film berdasarkan genre adalah model yang berbasis dengan konten atau biasa disebut _Content-Based Filtering_.
- Melakukan evaluasi terhadap metrik dari model _Machine Learning_ tersebut.

# _Data Understanding_

## Sumber Dataset
[Movie Recommender System Dataset](https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset)

## Jumlah Dataset
- Dataset movies.csv berjumlah 2731 baris dan 3 kolom
- Dataset ratings.csv berjumlah 100836 baris dan 4 kolom

## File Dataset
- movies.csv variabelnya terdiri dari
  - movieId: id film
  - title: Judul film
  - genres: genre film
 
- ratings.csv variabelnya terdiri dari
  - userId: id user
  - movieId: id film
  - rating: rating yang diberikan user
  - timestamp: waktu user memberikan rating

## Jenis dan Ukuran
Dataset berjenis zip dengan ukuran 866 kb
 
## _Exploratory Data Analysis - Univariate Analysis_
- _Movies_
  
  ![genre](https://github.com/lusiana02/MLRecommendationSystemFilmTitleBasedOnGenre/assets/123287899/c5c12840-19f7-4b11-8a4c-bcebb8807457)

  Gambar 1 Distribusi Fitur Genre
  
  Sebagian besar sampel film dari dataset movies ber-genre drama dan _comedy_, hal tersebut menunjukkan bahwa film yang tersedia lebih banyak ber-genre drama dan     _comedy_.

- Rating

  ![rating](https://github.com/lusiana02/MLRecommendationSystemFilmTitleBasedOnGenre/assets/123287899/8fd0ce9e-0b06-4bc6-9f6a-b6e7a98cc4bd)

  Gambar 2 Visualisasi fitur numerik rating

  Dari hasil visualisasi pada gambar 4 dapat disimpulkan bahwa:

  - Rentang rating film adalah 0,5 hingga 5
  - Jumlah sampel terbanyak adalah film yang memiliki rating 4, hal ini menunjukkan bahwa banyak _user_ yang menilai film dengan nilai 4.

# _Data Preparation_
Berikut merupakan tahapan-tahapan dalam melakukan _data preparation_:

- Menggabungkan Dataset dan Menangani _Missing Value_
  Proses ini dilakukan dengan menggabungkan kedua dataset movies.csv dan ratings.csv menggunakan fungsi merge() dan setelah digabungkan data yang memiliki nilai     kosong/null akan dihapus menggunakan fungsi _dropna()_ dengan tujuan agar mudah untuk diproses.
- Menghapus Data Duplikat
  Proses ini dilakukan dengan menggunakan fungsi _drop_duplicates()_ agar tidak ada data yang memiliki nilai sama untuk mencegah kekeliruan.
- Mengonversi Data Series Menjadi Bentuk List
  Proses ini dilakukan dengan menggunakan fungsi _tolist()_ agar data lebih mudah diproses pada tahap pemodelan.

# _Modeling_
_Content-Based Filtering_ adalah salah satu metode yang digunakan dalam sistem rekomendasi untuk memberikan saran kepada pengguna berdasarkan analisis konten dari item yang diminati oleh pengguna. Metode algoritma ini memanfaatkan karakteristik atau fitur yang ada pada item untuk mengidentifikasi kesamaan dan memberikan rekomendasi yang relevan.

Alasan menggunakan metode _Content-Based Filtering_ untuk masalah ini karena dapat memberikan rekomendasi yang baik bahkan untuk pengguna baru atau item baru yang belum memiliki interaksi dengan pengguna lain. Ini karena rekomendasi didasarkan pada konten dari item itu sendiri, bukan pada interaksi sebelumnya.

Setelah data selesai disiapkan, proses selanjutnya adalah membuat model adapun tahap-tahapnya diantaranya sebagai berikut:

- Melakukan Vektorisasi dengan TF-IDF
  Pada tahap ini data yang telah disiapkan dikonversi menjadi bentuk vektor menggunakan fungsi _tfidfvectorizer()_ dari _library sklearn_ untuk mengidentifikasi     korelasi antara judul film dengan kategori genrenya.
- Mengukur tingkat kesamaan dengan _Cosine Similarity_
  Setelah data dikonversi menjadi bentuk vektor, selanjutnya ukur tingkat kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama. Semakin kecil sudut _cosinus_, semakin besar nilai _cosine similarity_.
- Membuat Fungsi _movie_recommendations()_ Tahap terakhir dari proses modeling adalah membuat fungsi untuk mendapatkan hasil _top-N recommendation_, kali ini fungsinya dinamakan _movie_recommendations()_. Cara kerja dari fungsi ini yaitu menggunakan fungsi argpartition untuk mengambil sejumlah nilai k tertinggi dari _similarity data_ (dalam kasus ini: _dataframe cosine_sim_df_). Kemudian mengambil data dari bobot (tingkat kesamaan) tertinggi ke terendah. Data ini lalu dimasukkan ke dalam variabel _closest_. Berikutnya menghapus _movie_title_ yang dicari menggunakan fungsi _drop()_ agar tidak muncul dalam daftar rekomendasi. Penjelasan parameter dari fungsi _movie_recommendations()_ adalah sebagai berikut:
  - _movie_title_ : Judul film (index kemiripan dataframe) (str)
  - _similarity_data_ : Kesamaan _dataframe_ simetrik dengan judul film sebagai indeks dan kolom (object)
  - _items_ : Mengandung kedua nama dan fitur lainnya yang digunakan untuk mendefinisikan kemiripan (object)
  - k : Banyaknya jumlah rekomendasi yang diberikan (int)
 
# _Result_
Setelah model selesai dibuat, panggil model untuk menampilkan hasil rekomendasi, sebagai contoh kita gunakan judul film Piper (2016) untuk menguji model.
|     |	id	  |movie_title	|genre    |
|-----|-------|-------------|---------|
|2565	|160718	|Piper (2016)	|Animation|

Tabel 1 Informasi judul film uji

Dapat terlihat pada Tabel 1 bahwa film Piper (2016) merupakan film dengan genre _Animation_. Selanjutnya kita lihat rekomendasi film yang sesuai dengan genre yang sama dengan film tersebut.

|	|movie_title|	genre|
|-|-----------|------|
|0|	A Plasticine Crow (1981)|	Animation|
|1|	The Red Turtle (2016)|	Animation|
|2|	The Monkey King (1964)|	Animation|
|3|	Winter in Prostokvashino (1984)|	Animation|
|4|	Vacations in Prostokvashino (1980)|	Animation|
|5|	Garfield's Pet Force (2009)|	Animation|
|6|	Nasu: Summer in Andalusia (2003)|	Animation|
|7|	Three from Prostokvashino (1978)|	Animation|
|8|	Investigation Held by Kolobki (1986)|	Animation|
|9|	Fireworks, Should We See It from the Side or t...	|Animation|

Tabel 2 Hasil Rekomendasi

Seperti terlihat pada Tabel 2, model berhasil menampilkan rekomendasi film berdasarkan genrenya.

# _Evaluation_
Karena model yang digunakan untuk proyek kali ini adalah _Content-Based Filtering_, maka metrik yang cocok untuk digunakan adalah _Precision_. Secara matematis dapat dirumuskan sebagai berikut:

$$
\text{precision} = \frac{\text{of our recommendations that are relevant}}{\text{of items we recommended}}
$$

Berdasarkan hasil yang telah ditampilkan Tabel 2 pada bagian _Result_ dapat disimpulkan bahwa dari 10 judul film yang direkomendasikan, ada 10 film yang relevan oleh karena itu nilai _Precision_ dari model ini adalah 10/10 atau 100%.

# _Conclusion_
Setelah melalui proses yang panjang, mulai dari mempersiapkan dataset hingga melakukan evaluasi, akhirnya sistem rekomendasi dengan pendekatan _Machine Learning_ _Content-Based Filtering_ pun selesai dirancang dan hasilnya pun cukup memuaskan yaitu dari 10 judul film yang direkomendasikan, terdapat 10 film yang relevan dengan judul film yang diuji yang menandakan _precision_ dari model ini adalah 100%. Diharapkan dengan dirancangnya sistem rekomendasi ini, _traffic platform film streaming_ perusahaan dapat naik dengan signifikan.

# Referensi
[1] Firmahsyah, Firmahsyah dan Tiur Gantini. "Penerapan Metode Content-Based Filtering Pada Sistem Rekomendasi Kegiatan Ekstrakulikuler (Studi Kasus Di Sekolah ABC)." Jurnal Teknik Informatika dan Sistem Informasi, vol. 2, no. 3, 2016, doi:10.28932/jutisi.v2i3.548.

[2] Mudjiono, Yoyon (2020) Kajian semiotika dalam film. Jurnal Ilmu Komunikasi, 1 (1). pp. 125-138. ISSN 2088-981X; 2723-255
