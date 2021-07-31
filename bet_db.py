# A BET database exporter class
import sqlalchemy as sq
from sqlalchemy.orm import Session
from tomocatdb.data_model import *
import numpy as np
import json 

class BetdbExporter:

    def __init__(self, conn_string, bet_obj): # Creates the inital db engine from the conn string
        self.engine = sq.create_engine(conn_string)
        self.bet_analysis = bet_obj


    def push_to_db(self):
        
        material_name, reactor_sample, rs_layer_code = self._get_file_metadata()
        
        stmt_zeo = sq.select(Zeolites.internal_id).where(Zeolites.internal_id == material_name)  # Query statement to search for a zeolite material
        stmt_ex = sq.select(Extrudates.internal_id).where(Extrudates.internal_id == material_name)  # Query statement to search for an extrudate material
        stmt_sample = sq.select(ReactorSamples.layer_code).where(ReactorSamples.layer_code == rs_layer_code)  # Query statement to search for a reactor sample

        with Session(self.engine) as session:
            
            zeo = session.execute(stmt_zeo).first()  # Zeolite 
            ext = session.execute(stmt_ex).first()  # Extrudate
            reactor_samp = session.execute(stmt_sample).first()  # Sample
                
            analysis_date = self.bet_analysis.metadata['Meas. Time.'].split(' ')[0].split('/')[::-1]  # creating a well-formated date string
            analysis_date = '/'.join(analysis_date)
            
            def np_encoder(object):  # An encoder to convert numpy to float/int prior to json dump
                if isinstance(object, np.generic):
                    return object.item()

            bet_res_data = json.dumps(self.bet_analysis.BET_results, default=np_encoder) # Creating json dump of bet_results

            bet_anal = GasAdsorptionAnalysis(
                                    adsorptive=self.bet_analysis.metadata['Adsorptive'],
                                    measurment_temp=self.bet_analysis.metadata['Meas. Temp./K'],
                                    volume_adsorbed=self.bet_analysis.metadata['Vs/ml'],
                                    sample_weight=self.bet_analysis.metadata['Sample weight/g'],
                                    bet_area=self.bet_analysis.BET_results['bet_area'],
                                    bet_results_params = bet_res_data,
                                    creation_date=analysis_date,
                                    data_loc = self.bet_analysis.file
                                )
                                
            # Checking for a sample material.
            if reactor_samp:  # This takes priority over zeolite and extrudate
                bet_anal.reactor_sample_id = reactor_samp.layer_code
            elif zeo:
                bet_anal.zeolite_id = zeo.internal_id
            elif ext:
                bet_anal.extrudate_id = ext.internal_id
            else:
                raise AssertionError ("No parent entry (zeolite, extrudate, or reactor sample) found in database!")
                
            session.add(bet_anal)
            session.commit()
            
            session.close()
        return 

    def _get_file_metadata(self):
        
        split_sign = '_'

        material_id, reactor_sample, _ = self.bet_analysis.file.split('\\')[-1].split(split_sign)
        reactor_layer_code = material_id + split_sign + reactor_sample

        return material_id, reactor_sample, reactor_layer_code



