import pandas as pd
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter as xl
import pygaps as pg
from pygaps.parsing.csv import isotherm_to_csv
from pygaps.graphing.calc_graphs import bet_plot, roq_plot
from pygaps.graphing.isotherm_graphs import plot_iso
from pygaps.characterisation.area_bet import bet_transform, roq_transform, area_BET_raw
from pygaps.core.adsorbate import Adsorbate

import os
from scipy.constants import R

class BET:

    def __init__(self, file, limits=None):
        self.file = file
        self.adsorption_data, self.desorption_data, self.all_data = self.read_DAT()
        self.isotherm = self.create_isotherm()
        self.adsorbate = Adsorbate.find(self.isotherm.adsorbate)
        self.limits = limits
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
        file_name, _, _ = self._get_result_path()

        fig_iso, ax_iso = plt.subplots(figsize=[5.5, 4.5])
        ax_iso.set_title('Isotherm: ' + file_name, fontsize=14)

        plot_iso(self.isotherm, ax=ax_iso)
        fig_iso.tight_layout()

        return ax_iso

    def BET_analysis(self):

        bet_area, c_const, n_monolayer, p_monolayer, slope, intercept, minimum, maximum, corr_coef = area_BET_raw(
                            self.isotherm.pressure(branch='ads'),
                            self.isotherm.loading(
                                branch='ads',
                                loading_unit='mol',
                                loading_basis='molar'
                                ), 
                            cross_section=self.adsorbate.get_prop("cross_sectional_area"),
                            limits=self.limits)


        BET_results = {
                            'bet_area': np.round(bet_area, 2),
                            'c_const': np.round(c_const, 2),
                            'n_monolayer': np.round(n_monolayer, 4),
                            'p_monolayer': np.round(p_monolayer, 4),
                            'bet_slope': np.round(slope, 2),
                            'bet_intercept': np.round(intercept, 2),
                            'corr_coef': np.round(corr_coef, 5),
                            'pressure_range' : [
                                                np.round(self.isotherm.pressure(branch='ads')[minimum], 4),
                                                np.round(self.isotherm.pressure(branch='ads')[maximum], 4) 
                                                ],
                            'minimum' : minimum,
                            'maximum' : maximum,
                            'limits': [minimum, maximum]
                        }
        self.BET_results = BET_results
        return BET_results
            
    def plot_bet(self):
        file_name, _, _ = self._get_result_path()

        fig_bet, ax_bet = plt.subplots(figsize=[5.5, 4.5])
        _, _, n_monolayer, p_monolayer, slope, intercept, _, _, minimum, maximum, _ = self.BET_results.values()

        ax_bet = bet_plot(
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

        ax_bet.set_title('BET plot: ' + file_name, fontsize=14)
        fig_bet.tight_layout()

        return ax_bet

    def plot_roq(self):

        file_name, _, _ = self._get_result_path()

        fig_roq, ax_roq = plt.subplots(figsize=[5.5, 4.5])

        _, _, n_monolayer, p_monolayer, slope, intercept, _, _, minimum, maximum, _ = self.BET_results.values()

        ax_roq = roq_plot(
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
            ax=ax_roq
            )

        ax_roq.set_title('Rouquerol plot: ' + file_name, fontsize=14)
        fig_roq.tight_layout()

        return ax_roq

    ##### EXPORT METHODS

    def to_json(self, filepath='default'):

        _, result_file_name, result_path = self._get_result_path()

        if filepath == 'default':
            res_path = os.path.join('RESULTS', 'ISOTHERM_' + result_file_name + '.json')
            if os.path.isdir('RESULTS'):
                self.isotherm.to_json(res_path)
            else:
                os.mkdir('RESULTS')
                self.isotherm.to_json(res_path)

        else:
            self.isotherm.to_json(result_path)

    def to_csv(self, filepath='default'):

        _, result_file_name, result_path = self._get_result_path()

        if filepath == 'default':
            res_path = os.path.join('RESULTS', 'ISOTHERM_' + result_file_name + '.csv')
            if os.path.isdir('RESULTS'):
                isotherm_to_csv(self.isotherm, res_path)
            else:
                os.mkdir('RESULTS')
                isotherm_to_csv(self.isotherm, res_path)
        else:
            isotherm_to_csv(self.isotherm, result_path)

        return

    def to_excel(self, filepath='default'):

        ## Geeting file path
        _, result_file_name, result_path = self._get_result_path()

        if filepath == 'default':
            res_path = os.path.join('RESULTS', 'ISOTHERM_' + result_file_name + '.xlsx')
            if not os.path.isdir('RESULTS'):
                os.mkdir('RESULTS')
        
        writer = pd.ExcelWriter(res_path, engine='xlsxwriter')

        ## Creating isotherm sheet
        df_isotherm = []
        for branch in ['ads', 'des']:
            isotherm_points = {
                'pressure_{}'.format(branch) : self.isotherm.pressure(branch=branch),
                'loading_{}'.format(branch) : self.isotherm.loading(branch=branch),
            }
            df = pd.DataFrame(isotherm_points)
            df_isotherm.append(df)


        start_cols = [0, 2]
        for i, df in enumerate(df_isotherm):
            df.to_excel(writer, sheet_name='isotherm',
             startcol=start_cols[i], startrow=11, index=False)


        ## Creating BET sheet
        sheetname = 'BET'

        main_bet_points = {
            'pressure' : self.isotherm.pressure(branch='ads'),
            'bet_points' : bet_transform(
                self.isotherm.pressure(branch='ads'),
                self.isotherm.loading(
                         branch='ads',
                         loading_unit='mol',
                         loading_basis='molar')
                         )
        }

        df_main_bet_points = pd.DataFrame(main_bet_points)
        df_main_bet_points.to_excel(writer, sheet_name=sheetname,
             startcol=0, startrow=0, index=False)

        bet_minimum = self.BET_results['minimum']
        bet_maximum = self.BET_results['maximum']

        selected_bet_point = {
            'pressure_selected' : self.isotherm.pressure(branch='ads')[bet_minimum:bet_maximum],
            'bet_points_selected' : bet_transform(
                self.isotherm.pressure(branch='ads'),
                self.isotherm.loading(
                         branch='ads',
                         loading_unit='mol',
                         loading_basis='molar')
                         )[bet_minimum:bet_maximum]
        }
        
        df_selected_bet_point = pd.DataFrame(selected_bet_point)
        df_selected_bet_point.to_excel(writer, sheet_name=sheetname,
             startcol=2, startrow=0, index=False)

        bet_p_monolayer = self.BET_results['p_monolayer']
        bet_n_monolayer = self.BET_results['n_monolayer']

        bet_monolayer_point = {
            'pressure_monolayer' : [bet_p_monolayer],
            'bet_point_monolayer' : [bet_transform(bet_p_monolayer, bet_n_monolayer)]
        }

        df_monolayer_point = pd.DataFrame(bet_monolayer_point)
        df_monolayer_point.to_excel(writer, sheet_name=sheetname,
             startcol=4, startrow=0, index=False)

        worksheet = writer.sheets[sheetname]
        
        # Writing BET_results dictonary to excel sheet
        col_num = 7  # The start column number
        for key, value in self.BET_results.items(): # Convert non-list value to list
            if type(value) == list:  
                pass
            else:
                value = [value]
            worksheet.write(0, col_num, key)
            worksheet.write_column(1, col_num, value)
            col_num += 1

        # Plotting BET

        workbook = writer.book

        chart = workbook.add_chart({'type' : 'scatter'})

        rows_bet_mp, _ = df_main_bet_points.shape
        chart.add_series({
                                    'name': 'All Points',
                                    'categories': [sheetname, 1, 0, rows_bet_mp, 0],
                                    'values':  [sheetname, 1, 1, rows_bet_mp, 1],
                                    'marker': {
                                                'type': 'circle',
                                                'size': 9,
                                                'fill' : {'color': 'white'},
                                                'border': {'color': 'gray'},
                                                }
                                })
        row_bet_sel, _ = df_selected_bet_point.shape
        chart.add_series({
                                    'name': 'Selected Points',
                                    'categories': [sheetname, 1, 2, row_bet_sel, 2],
                                    'values':  [sheetname, 1, 3, row_bet_sel, 3],
                                    'marker': {
                                                'type': 'circle',
                                                'size': 9,
                                                'border': {'color': 'red'},
                                                }
                                })

        row_bet_mono, _ = df_monolayer_point.shape
        chart.add_series({
                                    'name': 'Monolayer Points',
                                    'categories' : [sheetname, 1, 4, row_bet_mono, 4],
                                    'values':  [sheetname, 1, 5, row_bet_mono, 5],
                                    'marker': {
                                                'type': 'square',
                                                'size': 9,
                                                'border': {'color': 'black'},
                                                }
                                })
        
        chart.set_x_axis({
                    'name': 'Pressure [p/p0]',
                    'num_format' : '0.00',
                    'min' : 0,
                    'max' : self.BET_results['pressure_range'][1]
            })

        chart.set_y_axis({
                'name': 'BET value',
                'min' : 0,
                'max' : 1.2*df_selected_bet_point.iloc[:, 1].max()
        })

        #chart.set_size({'x_scale': 1.5, 'y_scale': 2})

        worksheet.insert_chart('E10', chart)
  
        writer.save()

        return

    ### METHOD WHICH GETS THE RESULT PATH
    def _get_result_path(self):

        list_path = self.file.split('\\')
        result_folder = '\\'.join(list_path[:-1] + ['RESULTS'])
        file_name = list_path[-1].split('.')[0]
        result_file_name = file_name + '_result'

        if os.path.isdir(result_folder):
            result_path = os.path.join(result_folder, result_file_name)
        else:
            os.mkdir(result_folder)
            result_path = os.path.join(result_folder, result_file_name)

        return file_name, result_file_name, result_path
