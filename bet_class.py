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
            
            df.drop(columns=['No.', 'Unnamed: 5'], index=0, inplace=True)
            df['Pe/P0'] = np.round(df['Pe/kPa']/df['P0/kPa'], 4)

            columns = df.columns.to_list()
            new_column_order = [columns[-1]] + columns[:-1]
            df = df[new_column_order].copy()    
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
        
        # --- loading measurment metadata ---
        instrument_metadata = pd.read_csv(self.file, sep='\t', skiprows=3, nrows=28, names=['instrument', 'value'])
        instrument_metadata = instrument_metadata.loc[
                                instrument_metadata['instrument'].isin([
                                    'Adsorptive:',
                                    'Meas. Temp./K:',
                                    'Vs/ml:',
                                    'Sample weight/g:',
                                    'Date of measurement:',
                                    'Time of measurement:'
                                    ])
                                ]
        meta_keys = instrument_metadata.iloc[:, 0].to_list()
        meta_keys = [key[:-1] for key in meta_keys]
        meta_values = instrument_metadata.iloc[:, 1].to_list()

        self.metadata = {key: val for key, val in zip(meta_keys, meta_values)}  # metadata dictionary
        self.metadata['Meas. Time.'] = self.metadata['Date of measurement'] + ' ' + self.metadata['Time of measurement']
        for key in ['Date of measurement', 'Time of measurement']:
            del self.metadata[key]

        # --- loading measurment data ---
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
                        adsorbate = self.metadata['Adsorptive'],
                        temperature = str(int(float(self.metadata['Meas. Temp./K']))),
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
                            'p_select_min' : np.round(self.isotherm.pressure(branch='ads')[minimum], 4),
                            'p_select_max' : np.round(self.isotherm.pressure(branch='ads')[maximum], 4) ,
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

    def to_json(self, filepath=None):

        _, result_file_name, default_path = self._get_result_path()

        if not filepath:
            res_path = default_path + '.xlsx'
        else:
            res_path = os.path.join(filepath, result_file_name + '.xlsx')

        self.isotherm.to_json(res_path)

    def to_csv(self, filepath='default'):

        _, result_file_name, default_path = self._get_result_path()

        if not filepath:
            res_path = default_path + '.csv'
        else:
            res_path = os.path.join(filepath, result_file_name + '.csv')

        isotherm_to_csv(self.isotherm, res_path)
        
        return

    def to_excel(self, filepath=None):

        ##Getting file path
        _, result_file_name, default_path = self._get_result_path()

        if not filepath:
            res_path = default_path + '.xlsx'
        else:
            res_path = os.path.join(filepath, result_file_name + '.xlsx')
        
        # -- Instanciating the xlsx writer
        writer = pd.ExcelWriter(res_path, engine='xlsxwriter')  # Defining the xlsx writer
        workbook = writer.book  # Definint the workbook
        cell_format = workbook.add_format({'bold': True, 'italic': True}) # cell format "bold"
        cell_format_it = workbook.add_format({'italic': True}) # cell format "italic"

        # -- Creating isotherm sheet
        isotherm_sheet = 'Isotherm'
        df_isotherm = []
        for branch in ['ads', 'des']:
            isotherm_points = {
                'pressure_{}'.format(branch) : self.isotherm.pressure(branch=branch),
                'loading_{}'.format(branch) : self.isotherm.loading(branch=branch),
            }
            df = pd.DataFrame(isotherm_points)
            df_isotherm.append(df)

        isotherm_start_row = 11
        start_cols = [0, 2]
        for i, df in enumerate(df_isotherm):
            df.to_excel(writer, sheet_name=isotherm_sheet,
             startcol=start_cols[i], startrow=isotherm_start_row, index=False)
        
        # -- writing isotherm dictonary to excel sheet
        iso_dict = {}
        for r in self.isotherm.to_csv().split('\n')[:10]:
            key, val = r.split(',')
            iso_dict[key] = val
        worksheet_iso = writer.sheets[isotherm_sheet]

        row_num = 0  # The start column number
        for key, value in iso_dict.items(): # Convert non-list value to list
            worksheet_iso.write(row_num, 0, key, cell_format)
            worksheet_iso.write_string(row_num, 1, value)
            row_num += 1

        # -- plotting isotherm

        chart_isotherm = workbook.add_chart({'type' : 'scatter', 'subtype': 'smooth_with_markers'})

        rows_ads = self.isotherm.pressure(branch='ads').shape[0]
        rows_des = self.isotherm.pressure(branch='des').shape[0]

        chart_isotherm.add_series({
                                    'name': '{}-ads.'.format(str(self.isotherm.adsorbate)),
                                    'categories': [isotherm_sheet, isotherm_start_row+1, 0, isotherm_start_row+rows_ads+1, 0],
                                    'values':  [isotherm_sheet, isotherm_start_row+1, 1, isotherm_start_row+rows_ads+1, 1],
                                    'marker': {
                                                'type': 'circle',
                                                'size': 8,
                                                'fill' : {'color': '#144F82'},
                                                'border': {
                                                    'color': '#144F82',
                                                    'width' : 1.5},
                                                    },
                                                'line' : {'color' : '#144F82'}
                                })
        
        chart_isotherm.add_series({
                                    'name': '{}-des.'.format(str(self.isotherm.adsorbate)),
                                    'categories': [isotherm_sheet, 13, 2, isotherm_start_row+rows_des+1, 2],
                                    'values':  [isotherm_sheet, 13, 3, isotherm_start_row+rows_des+1, 3],
                                    'marker': {
                                                'type': 'circle',
                                                'size': 8,
                                                'fill' : {'color': 'white'},
                                                'border': {
                                                    'color': '#144F82',
                                                    'width' : 1.5},
                                                    },
                                                'line' : {
                                                    'color' : '#144F82',
                                                    'dash_type': 'dash'}
                                })

        chart_isotherm.set_x_axis({
                    'name': 'Pressure [p/p0]',
                    'name_font' : {'size' : 18},
                    'num_format' : '0.00',
                    'num_font':  {'size': 13},
                    'max' : 1,
                    'min' : 0,
            })

        chart_isotherm.set_y_axis({
                'name': 'Loading (ml/g [STP])',
                'name_font' : {'size' : 18},
                'num_font':  {'size': 14 },
                'major_gridlines': {'visible': False},
        })

        chart_isotherm.set_plotarea({
                'border': {'color': 'black', 'width': 1.25}
            })

        chart_isotherm.set_size({'x_scale': 1.45, 'y_scale': 1.6})

        worksheet_iso.insert_chart('H5', chart_isotherm)

        # -- Creating the BET sheet
        sheetname = 'BET'

        # -- full bet plot
        bet_start_row = 13

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
             startcol=0, startrow=bet_start_row, index=False)

        # -- selected bet plot from roq criteria
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
             startcol=2, startrow=bet_start_row, index=False)

        # -- selected monolayer point
        bet_p_monolayer = self.BET_results['p_monolayer']
        bet_n_monolayer = self.BET_results['n_monolayer']

        bet_monolayer_point = {
            'pressure_monolayer' : [bet_p_monolayer],
            'bet_point_monolayer' : [bet_transform(bet_p_monolayer, bet_n_monolayer)]
        }

        df_monolayer_point = pd.DataFrame(bet_monolayer_point)
        df_monolayer_point.to_excel(writer, sheet_name=sheetname,
             startcol=4, startrow=bet_start_row, index=False)
        
        # -- instanciating the bet worksheet
        worksheet = writer.sheets[sheetname]

        # -- writing BET_results dictonary to excel sheet
        row_num = 0  # The start column number
        for key, value in self.BET_results.items(): # Convert non-list value to list
            worksheet.write(row_num, 0, key, cell_format)
            if not type(value) == list:  
                value = [value]
            worksheet.write_row(row_num, 1, value)
            if key == 'bet_area':
                worksheet.write(row_num, 2, 'm^2/g', cell_format_it)
            if key == 'n_monolayer':
                worksheet.write_string(row_num, 2, 'mol/g', cell_format_it)
            row_num += 1

        # -- plotting BET
        chart = workbook.add_chart({'type' : 'scatter'})

        rows_bet_mp, _ = df_main_bet_points.shape
        chart.add_series({
                                    'name': 'All Points',
                                    'categories': [sheetname, bet_start_row+1, 0, bet_start_row+rows_bet_mp+1, 0],
                                    'values':  [sheetname, bet_start_row+1, 1, bet_start_row+rows_bet_mp+1, 1],
                                    'marker': {
                                                'type': 'circle',
                                                'size': 8,
                                                'fill' : {'color': 'white'},
                                                'border': {
                                                    'color': 'gray',
                                                    'width' : 1.5},
                                                }
                                })
        row_bet_sel, _ = df_selected_bet_point.shape
        chart.add_series({
                                    'name': 'Selected Points',
                                    'categories': [sheetname, bet_start_row+1, 2, bet_start_row+row_bet_sel+1, 2],
                                    'values':  [sheetname, bet_start_row+1, 3, bet_start_row+row_bet_sel+1, 3],
                                    'marker': {
                                                'type': 'circle',
                                                'size': 8,
                                                'fill': {'color': 'red'},
                                                'border' : {'color' : 'red'}
                                                },
                                    'trendline': {
                                        'type': 'linear',
                                        'name' : 'fit',
                                        'line': {
                                        'width': 1.2,
                                        'dash_type': 'long_dash'},
                                    },
                            })

        row_bet_mono, _ = df_monolayer_point.shape
        chart.add_series({
                                    'name': 'Monolayer Point',
                                    'categories' : [sheetname, bet_start_row+1, 4, bet_start_row+row_bet_mono+1, 4],
                                    'values':  [sheetname, bet_start_row+1, 5, bet_start_row+row_bet_mono+1, 5],
                                    'marker': {
                                                'type': 'x',
                                                'size': 8,
                                                'fill' : {'color' : 'black'},
                                                'border' : {'color' : 'black'}
                                                }
                                })
        
        chart.set_x_axis({
                    'name': 'Pressure (p/p0)',
                    'name_font' : {'size' : 18},
                    'num_format' : '0.00',
                    'num_font':  {'size': 13 },
                    'min' : 0,
                    'max' : self.BET_results['p_select_max']
            })

        chart.set_y_axis({
                'name': '1/[n(p0/p - 1)]',
                'name_font' : {'size' : 18},
                'num_font':  {'size': 13 },
                'min' : 0,
                'max' : 1.3*df_selected_bet_point.iloc[:, 1].max(),
                'major_gridlines': {'visible': False},
        })

        chart.set_plotarea({
                'border': {'color': 'black', 'width': 1.25}
            })

        chart.set_size({'x_scale': 1.45, 'y_scale': 1.6})
        worksheet.insert_chart('H5', chart)

        # -- Rouquerol

        roq_sheet = 'Rouquerol'

        roq_start_row = 0

        main_roq_points = {
            'pressure' : self.isotherm.pressure(branch='ads'),
            'roq_points' : roq_transform(
                self.isotherm.pressure(branch='ads'),
                self.isotherm.loading(
                         branch='ads',
                         loading_unit='mol',
                         loading_basis='molar')
                         )
        }

        df_main_roq_points = pd.DataFrame(main_roq_points)
        df_main_roq_points.to_excel(writer, sheet_name=roq_sheet,
             startcol=0, startrow=roq_start_row, index=False)

        # -- selected bet plot from roq criteria
        bet_minimum = self.BET_results['minimum']
        bet_maximum = self.BET_results['maximum']

        selected_roq_point = {
            'pressure_selected' : self.isotherm.pressure(branch='ads')[bet_minimum:bet_maximum],
            'roq_points_selected' : roq_transform(
                self.isotherm.pressure(branch='ads'),
                self.isotherm.loading(
                         branch='ads',
                         loading_unit='mol',
                         loading_basis='molar')
                         )[bet_minimum:bet_maximum]
        }
        
        df_selected_roq_point = pd.DataFrame(selected_roq_point)
        df_selected_roq_point.to_excel(writer, sheet_name=roq_sheet,
             startcol=2, startrow=roq_start_row, index=False)

        # -- selected monolayer point
        bet_p_monolayer = self.BET_results['p_monolayer']
        bet_n_monolayer = self.BET_results['n_monolayer']

        roq_monolayer_point = {
            'pressure_monolayer' : [bet_p_monolayer],
            'bet_point_monolayer' : [roq_transform(bet_p_monolayer, bet_n_monolayer)]
        }

        df_roq_monolayer_point = pd.DataFrame(roq_monolayer_point)
        df_roq_monolayer_point.to_excel(writer, sheet_name=roq_sheet,
             startcol=4, startrow=roq_start_row, index=False)

        # -- plotting roq plot
        chart_roq = workbook.add_chart({'type' : 'scatter'})

        rows_roq_mp, _ = df_main_roq_points.shape
        chart_roq.add_series({
                                    'name': 'All Points',
                                    'categories': [roq_sheet, roq_start_row+1, 0, roq_start_row+rows_roq_mp+1, 0],
                                    'values':  [roq_sheet, roq_start_row+1, 1, roq_start_row+rows_roq_mp+1, 1],
                                    'marker': {
                                                'type': 'circle',
                                                'size': 8,
                                                'fill' : {'color': 'white'},
                                                'border': {
                                                    'color': 'gray',
                                                    'width' : 1.5},
                                                }
                                })
        row_roq_sel, _ = df_selected_roq_point.shape
        chart_roq.add_series({
                                    'name': 'Selected Points',
                                    'categories': [roq_sheet, roq_start_row+1, 2, roq_start_row+row_roq_sel+1, 2],
                                    'values':  [roq_sheet, roq_start_row+1, 3, roq_start_row+row_roq_sel+1, 3],
                                    'marker': {
                                                'type': 'circle',
                                                'size': 8,
                                                'fill': {'color': 'red'},
                                                'border' : {'color' : 'red'}
                                                },
                            })

        row_roq_mono, _ = df_roq_monolayer_point.shape
        chart_roq.add_series({
                                    'name': 'Monolayer Point',
                                    'categories' : [roq_sheet, roq_start_row+1, 4, roq_start_row+row_roq_mono+1, 4],
                                    'values':  [roq_sheet, roq_start_row+1, 5, roq_start_row+row_roq_mono+1, 5],
                                    'marker': {
                                                'type': 'x',
                                                'size': 8,
                                                'fill' : {'color' : 'black'},
                                                'border' : {'color' : 'black'}
                                                }
                                })
        
        chart_roq.set_x_axis({
                    'name': 'Pressure (p/p0)',
                    'name_font' : {'size' : 18},
                    'num_format' : '0.00',
                    'num_font':  {'size': 13 },
                    'min' : 0,
                    'max' : 1.0
            })

        chart_roq.set_y_axis({
                'name': 'n(1 - p/p0)',
                'name_font' : {'size' : 18},
                'num_font':  {'size': 13 },
                'num_format' : '0.0E+00',
                'min' : 0,
                'major_gridlines': {'visible': False},
        })

        chart_roq.set_plotarea({
                'border': {'color': 'black', 'width': 1.25}
            })

        chart_roq.set_size({'x_scale': 1.45, 'y_scale': 1.6})

        roq_worksheet = writer.sheets[roq_sheet]
        roq_worksheet.insert_chart('H5', chart_roq)

        # -- Writing raw data

        raw_data_sheet = 'Raw Data'

        self.adsorption_data.to_excel(writer, sheet_name=raw_data_sheet,
             startcol=0, startrow=1, index=False)

        self.desorption_data.to_excel(writer, sheet_name=raw_data_sheet,
             startcol=5, startrow=1, index=False)

        raw_worksheet = writer.sheets[raw_data_sheet]
        raw_worksheet.write_string(0, 0, 'Adsorption', cell_format)
        raw_worksheet.write_string(0, 5, 'Desorption', cell_format)

        # -- writing BET_results dictonary to excel sheet
        row_num = 1  # The start column number
        raw_worksheet.write_string(0, 11, 'Measurment metadata', cell_format)
        for key, value in self.metadata.items(): # Convert non-list value to list
            raw_worksheet.write_string(row_num, 11, key, cell_format)
            raw_worksheet.write_string(row_num, 12, value)
            row_num += 1

        writer.save()
        return

    ### METHOD WHICH GETS THE RESULT PATH
    def _get_result_path(self):

        list_path = self.file.split('\\')
        default_result_folder = '\\'.join(list_path[:-1] + ['RESULTS'])
        file_name = list_path[-1].split('.')[0]
        result_file_name = file_name + '_result'

        if os.path.isdir(default_result_folder):
            result_path = os.path.join(default_result_folder, result_file_name)
        else:
            os.mkdir(default_result_folder)
            result_path = os.path.join(default_result_folder, result_file_name)

        return file_name, result_file_name, result_path

