# BET-Analysis

A dashboard for performing batch mode BET (Brunauer-Emmett-Teller) analysis from gas adsorption measurments. The dashboard is built on a python class wrapping the [pyGAPS](https://pygaps.readthedocs.io/en/master/#) package. With the [tomocatdb](https://github.com/NicHaaJun/TomocatDB) package installed it is possible to commit analysed data directly to a local postgres server.

The best way of using the BET_dashboard notebook is via the Voila dashboarding tool.

With Voila installed simply:

```
cd BET_dashboard
voila BET_dashboard.ipynb
```
