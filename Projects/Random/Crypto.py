# Description: This program gets the price of crypto currencies in real time

from bs4 import BeautifulSoup
import requests
import time


# Create a function to get the price of a cryptocurrency
def get_crypto_price(coin):
    # Get the URL
    url = "https://www.google.com/search?q=" + coin + "+price"

    # Make a request to the website
    HTML = requests.get(url)

    # Parse the HTML
    soup = BeautifulSoup(HTML.text, 'html.parser')

    # Find the current price
    # text = soup.find("div", attrs={'class':'BNeawe iBp4i AP7Wnd'}).text
    text = soup.find("div", attrs={'class': 'BNeawe iBp4i AP7Wnd'}).find("div",
                                                                         attrs={'class': 'BNeawe iBp4i AP7Wnd'}).text
    # Return the text
    return text


# Create a main function to show the price of the cryptocurrency
def main():
    # Set dummy for last price value
    last_price = -1
    last_price1 = -1
    last_price2 = -1

    # Create an infinite loop to show the price
    while True:
        # Choose cryptos
        crypto = 'bitcoin'
        crypto1 = 'litecoin'
        crypto2 = 'ethereum'

        # Retrieve prices
        price = get_crypto_price(crypto)
        price1 = get_crypto_price(crypto1)
        price2 = get_crypto_price(crypto2)

        # Check if the price has changed
        if price != last_price:
            print(crypto + ' price: ', price)
            last_price = price  # Update last price

        if price1 != last_price1:
            print(crypto1 + ' price: ', price1)
            last_price1 = price1

        if price2 != last_price2:
            print(crypto2 + ' price: ', price2)
            last_price2 = price2

        time.sleep(3)  # Pause execution for 3 seconds


main()

