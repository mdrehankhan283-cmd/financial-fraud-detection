from pathlib import Path
from sklearn.datasets import fetch_openml
if __name__ == '__main__':
    data = fetch_openml(data_id=1597, as_frame=True, parser='auto')
    frame = data.frame.rename(columns={data.target.name: 'Class'})
    frame.to_csv(Path(__file__).parent / 'creditcard.csv', index=False)
    print(f'Downloaded {len(frame):,} rows. Source: OpenML 1597')
