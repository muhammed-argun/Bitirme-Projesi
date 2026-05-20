import pandas as pd
import glob
import os

path = r"E:\Muhammed Özel\Bitirme Projesi\test_veri"
files = sorted(glob.glob(os.path.join(path, "*.xlsx")))

def urun_siniflandir(urun_adi):
    u = str(urun_adi).lower()
    if any(x in u for x in ["1. sınıf", "2. sınıf", "3. sınıf", "4. sınıf"]): return "İlkokul (1-4)"
    elif any(x in u for x in ["5. sınıf", "6. sınıf", "7. sınıf"]): return "Ortaokul (5-7)"
    elif "8. sınıf" in u or "lgs" in u: return "LGS Grubu"
    elif any(x in u for x in ["9. sınıf", "10. sınıf", "11. sınıf"]): return "Lise (9-11)"
    elif any(x in u for x in ["12. sınıf", "tyt", "ayt", "yks"]): return "YKS/Üniversite Hazırlık"
    elif any(x in u for x in ["kpss", "dgs", "ales"]): return "Sınav Hazırlık (Memurluk/Lisansüstü)"
    return "Diğer"

all_months_data = []
for file in files:
    date_str = os.path.basename(file).replace('.xlsx', '')
    df = pd.read_excel(file, engine='openpyxl')
    df['Sinif_Grubu'] = df['Ürün Adı'].apply(urun_siniflandir)
    df['Donem'] = date_str
    # Sadece gerekli sütunları alarak hafızayı rahatlatıyoruz
    all_months_data.append(df[['Sinif_Grubu', 'Donem', 'Net Satış Adedi']])

df_master = pd.concat(all_months_data)
ts_data = df_master.pivot_table(index='Sinif_Grubu', columns='Donem', values='Net Satış Adedi', aggfunc='sum').fillna(0)
# Sütunları tarihe göre (2021.09, 2021.10...) garantiye alalım
ts_data = ts_data.reindex(sorted(ts_data.columns), axis=1)
ts_data.to_csv("hazir_veri_seti_gruplanmis_tumu.csv")
print("Temiz veri seti hazır!")