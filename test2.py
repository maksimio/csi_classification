from re import I
from metawifi import MetaWifi

mf = MetaWifi('./csi/homelocation/five place')
print(mf.df_raw)
mf.df_raw.drop(['payload', 'csi', 'path'], axis=1).head(1000).to_csv('test.csv', index=False)


#TODO: метод, который добывает пути рекурсивно для всех файлов





























# --- ПРЕИМУЩЕСТВА ---

# Значительное ускорение и упрощение (только нативный python) кода + сделал его действительно универсальным (ускорение важно для raspberry)

# Простота кода + менеджмент (поле "time")

# Какой процент пакетов является мусором для каких категории и типа - дополнительный анализ данных

# Посторонние пакеты с CSI - интересно

# RSSI и другие метрики

# Возможность классификации временным окном (двумерная) благодаря timestamp - широкое поле для экспериментов

# Расширяемость - теперь точно все возможные данные берем