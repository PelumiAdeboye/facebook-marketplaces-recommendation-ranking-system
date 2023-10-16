import pandas as pd

def clean_price(price):
    # Remove pound sign and comma from price and convert to float
    return float(price.replace('£', '').replace(',', ''))

def clean_data(df):
    # Remove rows with null values
    df = df.dropna()

    # Convert prices to numerical format
    df['price'] = df['price'].apply(clean_price)

    return df

if __name__ == '__main__':
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv('Products.csv')

    # Clean the data
    cleaned_df = clean_data(df)

    # Save the cleaned DataFrame to a new CSV file
    cleaned_df.to_csv('cleaned_tabular_data.csv', index=False)
