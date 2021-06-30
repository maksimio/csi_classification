import pandas as pd
from metawifi import MetaWifi

mf = MetaWifi('./csi/homelocation/five place')

print(mf.df_raw)
# mf.df_raw.drop(['payload', 'csi', 'path'], axis=1).head(1000).to_csv('test.csv', index=False)

df1 = mf.df_raw.groupby(['type', 'category',]).count()['field_len']
df2 = mf.df_raw[mf.df_raw['csi_len'] == 0].groupby(['type', 'category',]).count()['csi_len']
df = pd.concat([df1, df2], axis=1)
df['wrong_%'] = (df2 / df1 * 100).astype(int)
print(df)
















































#TODO: метод, который добывает пути рекурсивно для всех файлов (и проверка на .dat в конце - пустые ридеры)





























# --- ПРЕИМУЩЕСТВА ---

# Значительное ускорение и упрощение (только нативный python) кода + сделал его действительно универсальным (ускорение важно для raspberry)

# Простота кода + менеджмент (поле "time" у классов)

# Какой процент пакетов является мусором для каких категории и типа - дополнительный анализ данных

# Посторонние пакеты с CSI - интересно

# RSSI и другие метрики

# Возможность классификации временным окном (двумерная) благодаря timestamp - широкое поле для экспериментов

# Расширяемость - теперь точно все возможные данные берем