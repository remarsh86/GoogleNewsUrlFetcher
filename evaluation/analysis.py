
import pandas as pd
import os
import config

if __name__ == '__main__':
    data = pd.read_csv(os.path.join(config.get_app_root(), "evaluation", 'politics_topics_old.csv'))
    print(data)
    # data.plot(x=data.index, y='bias')
    print("123")
