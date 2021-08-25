# BET-Analysis

A dashboard for performing batch mode BET (Brunauer-Emmett-Teller) analysis from gas adsorption measurments. The dashboard is built on a python class wrapping the [pyGAPS](https://pygaps.readthedocs.io/en/master/#) package. With the [tomocatdb](https://github.com/NicHaaJun/TomocatDB) package installed it is possible to commit analysed data directly to a local postgres server.

The best way of using the BET_dashboard notebook is via the [Voila](https://voila.readthedocs.io/en/stable/index.html) dashboarding tool.

With Voila installed simply:

```
cd BET_dashboard
voila BET_dashboard.ipynb
```
![image](https://user-images.githubusercontent.com/70808555/130829766-648c1149-91e0-402e-83c3-3272f635653c.png)

## Database Connection





