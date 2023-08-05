import yfinance as yf
import matplotlib.pyplot as plt

from flask import Flask, request, Response

app = Flask(__name__)

def fetch_chart(ticker, period='1mo', interval='1d', technical_indicator='Close'):
    try:
        stock_data = yf.download(ticker, period=period, interval=interval)
        stock_data[technical_indicator].plot(title=f'{ticker} - {technical_indicator} ({period})')
        plt.xlabel('Date')
        plt.ylabel(technical_indicator)
        plt.grid()
        plt.tight_layout()
        plt.savefig('chart.png')
        plt.close()

        with open('chart.png', 'rb') as file:
            chart_bytes = file.read()

        return chart_bytes

    except Exception as e:
        return str(e)

@app.route('/get_chart', methods=['GET'])
def get_chart():
    ticker = request.args.get('ticker')
    period = request.args.get('period', '1mo')
    interval = request.args.get('interval', '1d')
    technical_indicator = request.args.get('indicator', 'Close')

    if not ticker:
        return Response('Ticker not provided.', status=400)

    chart_bytes = fetch_chart(ticker, period, interval, technical_indicator)
    if isinstance(chart_bytes, str):
        return Response(chart_bytes, status=500)

    return Response(chart_bytes, status=200, content_type='image/png')

if __name__ == '__main__':
    app.run(debug=True)
