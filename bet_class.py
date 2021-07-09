import pandas as pd
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
import pygaps as pg
from pygaps.graphing.calc_graphs import bet_plot, roq_plot
from pygaps.characterisation.area_bet import bet_transform, roq_transform, area_BET_raw
from pygaps.core.adsorbate import Adsorbate

import os
from scipy.constants import R

class BET:

    def __init__(self, file):
        self.file = file
        self.adsorption_data, self.desorption_data, self.all_data = self.read_DAT()
        self.isotherm = self.create_isotherm()
        self.adsorbate = Adsorbate.find(self.isotherm.adsorbate)
        self.BET_results = self.BET_analysis()

    def read_DAT(self):

        def clean_df(df):
            
            volume_to_mols = lambda x: x*1e-6*100e3/(273.15*R)  # Calculating mass
            STP_to_real = lambda x: x/(273.15/77.36)

            df.drop(columns=['No.', 'Unnamed: 5'], index=0, inplace=True)
            df['Pe/P0'] = df['Pe/kPa']/df['P0/kPa']

            columns = df.columns.to_list()
            new_column_order = [columns[-1]] + columns[:-1]
            df = df[new_column_order].copy()
            
            df['V/real mlg-1'] = STP_to_real(df['V/ml(STP) g-1'])
            
            return df

        # Here we look for the line no. definint adsorption and desorption data
        with open(self.file, 'r') as f:
            id_ads = []
            id_des = []
            i = 0
            for line in f:
                if line.strip() == 'Adsorption data':
                    id_ads.append(i+2)  # We skip two lines
                elif line.strip() == 'Desorption data':
                    id_ads.append(i-2) # We subtract two lines
                    id_des.append(i+2) # We skip to lines

                i += 1
        
        
        no_ads_rows = id_ads[1] - id_ads[0] #  Number of lines spannind the adsorption data
        adsorption_data = pd.read_csv(self.file, sep='\t', skiprows=id_ads[0], nrows=no_ads_rows-1, engine='python')
        adsorption_data = clean_df(adsorption_data)
        
        desorption_data = pd.read_csv(self.file, sep='\t', skiprows=id_des[0], skipfooter=1, engine='python')
        desorption_data = clean_df(desorption_data)

        all_data = adsorption_data.append(desorption_data, ignore_index=True)

        return adsorption_data, desorption_data, all_data

    def create_isotherm(self):

        isotherm = pg.PointIsotherm(
                        pressure = self.all_data['Pe/P0'].to_numpy(),
                        loading = self.all_data['V/ml(STP) g-1'].to_numpy(),
                        
                        pressure_mode = 'relative',
                        pressure_unit = 'kPa',
                        loading_basis = 'molar',
                        loading_unit = 'cm3(STP)',
                        
                        material = 'zeolite',
                        material_basis = 'mass',
                        material_unit = 'g',
                        adsorbate = 'nitrogen',
                        temperature = '77',
                        temperature_unit = 'K'
                        
                    )
        return isotherm

    def plot_isotherm(self):
        return

    def BET_analysis(self):

        bet_area, c_const, n_monolayer, p_monolayer, slope, intercept, minimum, maximum, corr_coef = area_BET_raw(
                            self.isotherm.pressure(branch='ads'),
                            self.isotherm.loading(
                                branch='ads',
                                loading_unit='mol',
                                loading_basis='molar'
                                ), 
                            cross_section=self.adsorbate.get_prop("cross_sectional_area"))


        BET_results = {
                            'area': bet_area,
                            'c_const': c_const,
                            'n_monolayer': n_monolayer,
                            'p_monolayer': p_monolayer,
                            'bet_slope': slope,
                            'bet_intercept': intercept,
                            'corr_coef': corr_coef,
                            'pressure_range' : [
                                                self.isotherm.pressure(branch='ads')[minimum],
                                                self.isotherm.pressure(branch='ads')[maximum] 
                                                ],
                            'minimum' : minimum,
                            'maximum' : maximum,
                            'limits': [minimum, maximum]
                        }
        return BET_results
            
    def plot_bet(self):

        fig_bet, ax_bet = plt.subplots(figsize=[4, 4])
        _, _, n_monolayer, p_monolayer, slope, intercept, _, minimum, maximum, _, _  = self.BET_results.values()

        bet_plot_ax = bet_plot(
            self.isotherm.pressure(branch='ads'),
            bet_transform(
                self.isotherm.pressure(branch='ads'),
                self.isotherm.loading(
                         branch='ads',
                         loading_unit='mol',
                         loading_basis='molar')
                         ),
            minimum, maximum, slope, intercept, p_monolayer,
            bet_transform(p_monolayer, n_monolayer),
            ax=ax_bet
        )

        fig_bet.tight_layout()

        return bet_plot_ax

    def plot_roq(self):

        fig, ax = plt.subplots(figsize=[4, 4])

        _, _, n_monolayer, p_monolayer, slope, intercept, _, minimum, maximum, _, _  = self.BET_results.values()

        roq_plot_ax = roq_plot(
            self.isotherm.pressure(branch='ads'),
            roq_transform(
                self.isotherm.pressure(branch='ads'),
                self.isotherm.loading(
                         branch='ads',
                         loading_unit='mol',
                         loading_basis='molar')
                         ),
            minimum, maximum, p_monolayer,
            roq_transform(p_monolayer, n_monolayer),
            ax=ax
            )
        fig.tight_layout()

        return roq_plot_ax


    def to_json(self, filepath='default'):

        list_path = self.file.split('\\')

        result_folder = '\\'.join(list_path[:-1] + ['RESULTS'])
        result_file_name = list_path[-1].split('.')[0] + '_result'

        def result_path(result_folder, result_file_name):

            if os.path.isdir(result_folder):
                return os.path.join(result_folder, result_file_name)
            else:
                os.mkdir(result_folder)
                return os.path.join(result_folder, result_file_name)
        
        if filepath == 'default':

            if os.path.isdir('RESULTS'):
                self.isotherm.to_json(os.path.join('RESULTS', result_file_name))
            else:
                os.mkdir('RESULTS')
                self.isotherm.to_json(os.path.join('RESULTS', result_file_name))

        else:
            self.isotherm.to_json(result_path(result_folder, result_file_name))

        return pg.isotherm_to_json(self.isotherm)

