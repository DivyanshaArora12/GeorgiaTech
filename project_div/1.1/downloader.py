from sec_edgar_downloader import Downloader

dl = Downloader("My Company","example@email.com","C:/Users/akshi/Downloads/project_div/1.1/result")

ticker = ["GOOGL", "WMT","MC"]

for tick in ticker:
    dl.get("10-K", tick, after="1995-01-01", before="2024-01-01")
