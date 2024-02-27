import json
import chardet
import streamlit as st
import pandas as pd
from llama_index.core.tools.tool_spec.base import BaseToolSpec

class ScriptureDescriptionToolSpec(BaseToolSpec):
    '''
    Purpose: Obtains the description or summary about vedas, mandalas, kandas, shuktas, archakah, adhyaya, and other scriptural elements.
    Returns: A dictionary containing the description or basic information about the specified scriptural element.
    Sample query:
    1. Describe the first kandah, second shukta from Atharvaveda?
    2. Summarize ShuklaYajurVeda?
    3. What is the difference between ShuklaYajurVeda and KrishnaYajurVeda?
    '''
    # Define the functions that we export to the LLM
    spec_functions = ["get_description"]

    with open("Data/scripture_descriptions.csv", 'rb') as f:
        result = chardet.detect(f.read())

    encoding = result['encoding']
    df = pd.read_csv("Data/scripture_descriptions.csv", encoding=encoding)

    @st.cache_data
    def get_description(_self, level_0, level_1:int=None, level_2:int=None, level_3:int=None):
        """
        To get the description or basic information about vedas/mandalas/kandas/shukatas/archakah/adhyaya and others.
        """
        try:
            if level_3 is not None:
                # Case with Level-2 specified
                result = _self.df[(_self.df['scripture_name'].str.lower() == level_0.lower())
                                  & (_self.df['level_1'] == str(level_1))
                                  & (_self.df['level_2'] == str(level_2)) & (_self.df['level_3'] == str(level_3))]
            elif level_2 is not None:
                # Case with Level-2 specified
                result = _self.df[(_self.df['scripture_name'].str.lower() == level_0.lower())
                                  & (_self.df['level_1'] == str(level_1)) & (_self.df['level_2'] == str(level_2))]
            elif level_1 is not None:
                # Case with Level-1 specified
                result = _self.df[(_self.df['scripture_name'].str.lower() == level_0.lower())
                                  & (_self.df['level_1'] == str(level_1))]
            else:
                # Case with only Level-0 specified
                result = _self.df[_self.df['scripture_name'].str.lower() == level_0.lower()]

            return result.iloc[0].to_dict()
        except IndexError as e:
            return json.dumps({"error": f"Failed to get scripture description. {e}"})

class MantraToolSpec(BaseToolSpec):
    '''
    To obtain translations or meaning of vedamantras from RigVeda and AtharvaVeda using the function `get_translation`.
    The mantra details such as vedamantra, padapatha, rishi, chandah, devata, and swarah from the vedas accessible through the function `get_vedamantra_details`.
    The mantra summary like anvaya, adhibautic, ahyatmic, adhidaivic meaning of vedamantra accessible using the function 'get_vedamantra_summary'
    Sample Query:
    1. What is the vedamantra of the mantra from Rigveda, first mandala, first shukta, and first mantra?
    2. What is the devata of the vedamantra from Rigveda, first mandala, first shukta, and first mantra?
    3. What is the meaning of the vedamantra from Rigveda, first mandala, first shukta, and first mantra written by Tulsi Ram?
    4. What is the (adhibautic) meaning of the vedamantra from RigVeda, first mandala, first shukta, and first mantra?
    '''
    spec_functions = ["get_translation", "get_vedamantra_details", "get_vedamantra_summary"]

    TRANSLATION_CSV_PATH = 'Data/trans_Rig_Ath_index_v2.csv'
    VEDAMANTRA_CSV_PATH = "Data/veda_content_modified_v3.csv"

    def __init__(self):
        super().__init__()
        self.df_translation = pd.read_csv(self.TRANSLATION_CSV_PATH, encoding='utf-8')
        self.df_vedamantra = pd.read_csv(self.VEDAMANTRA_CSV_PATH, encoding='utf-8')

    @st.cache_data
    def get_translation(_self, mantraid=None, scripture_name=None, MahatmaName=None, KandahNumber=None,
                        MandalaNumber=None, ArchikahNumber=None, ShuktaNumber=None,
                        AnvayaNumber=None, PrapatakNumber=None, MantraNumber=None,
                        AnuvakNumber=None, AdhyayaNumber=None):
        """
        Get the translation of mantras from RigVeda and AtharvaVeda.
        Sample Query:
        1. What is the translation of Tulsi Ram of the vedamantra from Rigveda, first mandala, first shukta, and first mantra?
        2. What is the translation or adhibautic meaning of the vedamantra from RigVeda, first mandala, first shukta, and first mantra?
        3. What is the subject of the mantra 1.1.84.1?
        """
        try:
            if mantraid is None:
                scripture_name_lower = scripture_name.lower() if scripture_name is not None else False
                if scripture_name_lower == 'rigveda':
                    details = _self.df_translation[
                        (_self.df_translation['scripture_name'].str.lower() == scripture_name_lower)
                        & (_self.df_translation['MandalaNumber'] == MandalaNumber)
                        & (_self.df_translation['ShuktaNumber'] == int(ShuktaNumber))
                        & (_self.df_translation['MantraNumber'] == int(MantraNumber))
                    ].to_dict(orient='records')
                elif scripture_name_lower == 'atharvaveda':
                    details = _self.df_translation[
                        (_self.df_translation['scripture_name'].str.lower() == scripture_name_lower)
                        & (_self.df_translation['KandahNumber'] == KandahNumber)
                        & (_self.df_translation['ShuktaNumber'] == ShuktaNumber)
                        & (_self.df_translation['MantraNumber'] == MantraNumber)].to_dict(orient='records')
                elif scripture_name_lower == 'samaveda':
                    details = _self.df_translation[
                        (_self.df_translation['scripture_name'].str.lower() == scripture_name_lower)
                        & (_self.df_translation['ArchikahNumber'] == ArchikahNumber)
                        & (_self.df_translation['ShuktaNumber'] == ShuktaNumber)
                        & (_self.df_translation['MantraNumber'] == MantraNumber)].to_dict(orient='records')
                elif scripture_name_lower == 'krishnayajurveda':
                    details = _self.df_translation[
                        (_self.df_translation['scripture_name'].str.lower() == scripture_name_lower)
                        & (_self.df_translation['PrapatakNumber'] == PrapatakNumber)
                        & (_self.df_translation['AnuvakNumber'] == AnuvakNumber)
                        & (_self.df_translation['MantraNumber'] == MantraNumber)].to_dict(orient='records')
                else:
                    details = _self.df_translation[
                        (_self.df_translation['scripture_name'].str.lower() == scripture_name_lower)
                        & (_self.df_translation['AdhyayaNumber'] == AdhyayaNumber)
                        & (_self.df_translation['MantraNumber'] == MantraNumber)
                    ].to_dict(orient='records')
            else:
                details = _self.df_translation[_self.df_translation['mantra_id'] == mantraid].to_dict(orient='records')

            if MahatmaName is not None:
                for item in details:
                    if item['MahatmaName'] == MahatmaName:
                        return item
            else:
                return details
        except Exception as e:
            return json.dumps({"error": f"Failed to get translation. {e}"})

    @st.cache_data
    def get_vedamantra_details(_self, mantraid=None, scripture_name=None, KandahNumber=None,
                               MandalaNumber=None, ArchikahNumber=None, ShuktaNumber=None,
                               AnvayaNumber=None, PrapatakNumber=None, MantraNumber=None,
                               AnuvakNumber=None, AdhyayaNumber=None):
        """
        To obtain the vedamantra details such as vedamantra, padapata, devata, rishi, swarah, and chandah.
        1. What is the vedamantra of the mantra from Rigveda, first mandala, first shukta, and first mantra?
        2. What is the devata of the vedamantra from Rigveda, first mandala, first shukta, and first mantra?
        """
        try:
            if mantraid is None:
                scripture_name_lower = scripture_name.lower() if scripture_name is not None else False

                if scripture_name_lower == 'rigveda':
                    conditions = (_self.df_vedamantra['scripture_name'].str.lower() == scripture_name_lower) & \
                                  (_self.df_vedamantra['MandalaNumber'] == MandalaNumber) & \
                                  (_self.df_vedamantra['ShuktaNumber'] == ShuktaNumber) & \
                                  (_self.df_vedamantra['MantraNumber'] == str(MantraNumber))
                    details = _self.df_vedamantra[conditions]['mantra_json'].values
                    vedamantra_details = json.loads(details[0])['mantraHeader']['language'][1]['mandala']['shukta']['mantra']

                elif scripture_name_lower == 'atharvaveda':
                    conditions = (_self.df_vedamantra['scripture_name'].str.lower() == scripture_name_lower) & \
                                  (_self.df_vedamantra['KandahNumber'] == KandahNumber) & \
                                  (_self.df_vedamantra['ShuktaNumber'] == ShuktaNumber) & \
                                  (_self.df_vedamantra['MantraNumber'] == str(MantraNumber))
                    details = _self.df_vedamantra[conditions]['mantra_json'].values
                    vedamantra_details = json.loads(details[0])['mantraHeader']['language'][1]['kandah']['shukta']['mantra']
                elif scripture_name_lower == 'samaveda':
                    conditions = (_self.df_vedamantra['scripture_name'].str.lower() == scripture_name_lower) & \
                                  (_self.df_vedamantra['ArchikahNumber'] == ArchikahNumber) & \
                                  (_self.df_vedamantra['ShuktaNumber'] == ShuktaNumber) & \
                                  (_self.df_vedamantra['MantraNumber'] == str(MantraNumber))
                    details = _self.df_vedamantra[conditions]['mantra_json'].values
                    vedamantra_details = json.loads(details[0])['mantraHeader']['language'][1]['archikah']
                elif scripture_name_lower == 'krishnayajurveda':
                    conditions = (_self.df_vedamantra['scripture_name'].str.lower() == scripture_name_lower) & \
                                  (_self.df_vedamantra['PrapatakNumber'] == PrapatakNumber) & \
                                  (_self.df_vedamantra['AnuvakNumber'] == AnuvakNumber) & \
                                  (_self.df_vedamantra['MantraNumber'] == str(MantraNumber))
                    details = _self.df_vedamantra[conditions]['mantra_json'].values
                    vedamantra_details = json.loads(details[0])['mantraHeader']['language'][1]['kandah']['prapatak']['anuvak']
                else:
                    conditions = (_self.df_vedamantra['scripture_name'].str.lower() == scripture_name_lower) & \
                                  (_self.df_vedamantra['AdhyayaNumber'] == AdhyayaNumber) & \
                                  (_self.df_vedamantra['MantraNumber'] == str(MantraNumber))
                    details = _self.df_vedamantra[conditions]['mantra_json'].values
                    vedamantra_details = json.loads(details[0])['mantraHeader']['language'][1]['adhyaya']['mantra']

            else:
                # Handle case when mantraid is provided
                details = _self.df_vedamantra[_self.df_vedamantra['mantra_number'] == mantraid]['mantra_json'].values
                vedamantra_details = json.loads(details[0])['mantraHeader']['language'][1]

            return vedamantra_details
        except Exception as e:
            return json.dumps({"error": f"Failed to get vedamantra details. {str(e)}"})

    @st.cache_data
    def get_vedamantra_summary(_self, mantraid=None, scripture_name=None, KandahNumber=None,
                               MandalaNumber=None, ArchikahNumber=None, ShuktaNumber=None,
                               AnvayaNumber=None, PrapatakNumber=None, MantraNumber=None,
                               AnuvakNumber=None, AdhyayaNumber=None):
        '''
        To obtain the vedamantra summary like anvaya, translation, adhibautic, adhyatmic, adhidaivic meaning of the mantra.
        1. What is the adhibautic meaning of the mantra from AtharvaVeda, first kandah, first shukta, and first mantra?
        2. What is the anvaya of the vedamantra from Rigveda, first mandala, first shukta, and first mantra?
        '''
        try:
            if mantraid is None:
                scripture_name_lower = scripture_name.lower() if scripture_name is not None else False
                if scripture_name_lower == 'rigveda':
                    details = _self.df_vedamantra[
                        (_self.df_vedamantra['scripture_name'].str.lower() == scripture_name_lower)
                        & (_self.df_vedamantra['MandalaNumber'] == MandalaNumber)
                        & (_self.df_vedamantra['ShuktaNumber'] == ShuktaNumber)
                        & (_self.df_vedamantra['MantraNumber'] == str(MantraNumber))
                    ]['mantra_json'].values
                elif scripture_name_lower == 'atharvaveda':
                    details = _self.df_vedamantra[
                        (_self.df_vedamantra['scripture_name'].str.lower() == scripture_name_lower)
                        & (_self.df_vedamantra['KandahNumber'] == KandahNumber)
                        & (_self.df_vedamantra['ShuktaNumber'] == ShuktaNumber)
                        & (_self.df_vedamantra['MantraNumber'] == str(MantraNumber))
                    ]['mantra_json'].values
                elif scripture_name_lower == 'samaveda':
                    details = _self.df_vedamantra[
                        (_self.df_vedamantra['scripture_name'].str.lower() == scripture_name_lower)
                        & (_self.df_vedamantra['ArchikahNumber'] == ArchikahNumber)
                        & (_self.df_vedamantra['ShuktaNumber'] == ShuktaNumber)
                        & (_self.df_vedamantra['MantraNumber'] == str(MantraNumber))
                    ]['mantra_json'].values
                elif scripture_name_lower == 'krishnayajurveda':
                    details = _self.df_vedamantra[
                        (_self.df_vedamantra['scripture_name'].str.lower() == scripture_name_lower)
                        & (_self.df_vedamantra['PrapatakNumber'] == PrapatakNumber)
                        & (_self.df_vedamantra['AnuvakNumber'] == AnuvakNumber)
                        & (_self.df_vedamantra['MantraNumber'] == str(MantraNumber))
                    ]['mantra_json'].values
                else:
                    details = _self.df_vedamantra[
                        (_self.df_vedamantra['scripture_name'].str.lower() == scripture_name_lower)
                        & (_self.df_vedamantra['AdhyayaNumber'] == AdhyayaNumber)
                        & (_self.df_vedamantra['MantraNumber'] == str(MantraNumber))
                    ]['mantra_json'].values
            else:
                details = _self.df_vedamantra[_self.df_vedamantra['mantra_number'] == mantraid]['mantra_json'].values

            jsonDict = json.loads(details[0])
            mantraSummary = jsonDict['mantraSummary']['language']
            mantraSummary_IAST = jsonDict['mantraSummary']['language'][1]
            vedamantra_summary = {"Roman-IAST summary of vedamantra": mantraSummary_IAST}
            for item in mantraSummary:
                if item['languageName'] == 'English':
                    vedamantra_summary.update({"English summary of vedamantra": item})
            return vedamantra_summary
        except Exception as e:
            return json.dumps({"error": f"Failed to get vedamantra summary. {e}"})
