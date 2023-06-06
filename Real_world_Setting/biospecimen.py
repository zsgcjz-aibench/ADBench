import csv
import sys
import os
import pandas as pd

bio_path='/mnt/sdb/pwb/StudyData/Biospecimen/BiospecimenResults/tmp_table/'

save_path='/mnt/sdb/pwb/bio/'

save_path_with_duplicates='/mnt/sdb/pwb/bio/9_tmp/'

Csf_Path = save_path+'7_tmp/csf/'
Save_Csf_Path = save_path+'8_tmp'
Save_Csf_Name = r'csf.csv'

Urine_Path = save_path+'7_tmp/urine/'
Save_Urine_Path = save_path+'8_tmp'
Save_Urine_Name = r'urine.csv'

Plasma_Path =save_path+ '7_tmp/plasma/'
Save_Plasma_Path = save_path+'8_tmp'
Save_Plasma_Name = r'plasma.csv'

Serum_Path =save_path+'7_tmp/serum/'
Save_Serum_Path = save_path+'8_tmp'
Save_Serum_Name = 'serum.csv'

viscode_2_viscode2_dic_path='/mnt/sdb/pwb/StudyData/Enrollment/Enrollment/ADNI2_VISITID.csv'

def get_viscode_2_viscode2_dic():
    map_dic={}
    with open(viscode_2_viscode2_dic_path, 'r') as tmp_f:
        tmp_array = tmp_f.readlines()
        for i in range(1, len(tmp_array)):
            sample_data=tmp_array[i].rstrip('\n').split(',')
            rid=eval(sample_data[0])
            viscode=eval(sample_data[1])
            viscode2=sample_data[2]

            if rid not in map_dic:
                map_dic[rid]={}
            tmp_sample=map_dic[rid]
            tmp_sample[viscode]=viscode2
            map_dic[rid]=tmp_sample
        return map_dic


def transform_viscode_to_viscode2(df):
    df.dropna(subset=['VISCODE'], inplace=True)
    map_dic=get_viscode_2_viscode2_dic()
    for index_order, row in df.iterrows():
        rid = eval(row['RID'])
        viscode=row['VISCODE']
        if 'v' in viscode:
            df.ix[index_order,'VISCODE']=map_dic[rid][viscode]

def print_first_line():
    with open(sys.argv[1], 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, rows in enumerate(reader):
            if i == 0:
                row = rows  # ADNIMERGE.csv first line
    print(row)

def keep_rows(csvname, keep_row, newcsvname):
    f=pd.read_csv(csvname, dtype=str)
    new_f=f[keep_row]
    new_f=new_f[~new_f['RID'].isin(["999999"])]      # ~ and delete the lines including 999999
    if newcsvname == Csf_Path+'ADNI_MESOSCALE.csv' :
        new_f.rename(columns={'TAU': 'TAU_1'}, inplace=True)
    if newcsvname == Csf_Path+'UPENNBIOMK9_04_19_17.csv' :
        new_f.rename(columns={'TAU': 'TAU_2','ABETA':'ABETA_2','PTAU':'PTAU_2'}, inplace=True)
    if newcsvname == Csf_Path+'UPENNBIOMKADNIDIAN2017.csv':
        new_f.rename(columns={'TAU': 'TAU_3','ABETA':'ABETA_3','PTAU':'PTAU_3'}, inplace=True)
    if newcsvname == Csf_Path+'UPENNBIOMK_MASTER.csv':
        new_f.rename(columns={'TAU': 'TAU_4','ABETA':'ABETA_4','PTAU':'PTAU_4'}, inplace=True)

    path = os.path.dirname(newcsvname)
    if not os.path.exists(path):
        os.makedirs(path)
    new_f = new_f.drop_duplicates()
    new_f.to_csv(newcsvname, index = False)

def drop_col(df, cutoff=0.05):
    n = len(df)
    cnt = df.count()
    cnt = cnt / n
    return df.loc[:, cnt[cnt >= cutoff].index]

def drop_row(df, cutoff=0.05) :
    n = df.shape[1]
    cnt = df.count(axis = 1)
    cnt = cnt / n
    return df.loc[cnt[cnt >= cutoff].index, :]

def fill_up_viscode2(Folder_Path) :
    file_list = os.listdir(Folder_Path)
    for i in range(0, len(file_list)):
        df = pd.read_csv(Folder_Path + '/' + file_list[i], dtype=str)
        if 'VISCODE2' in df.columns:
            df.dropna(subset=['VISCODE2'], inplace=True)
            if 'VISCODE' in df.columns:
                df = df.drop(['VISCODE'], axis=1)
            if 'EXAMDATE' in df.columns:
                df = df.drop(['EXAMDATE'], axis=1)
            df.to_csv(Folder_Path + '/' + file_list[i])
            pass
        else :
            if 'VISCODE' in df.columns:
                transform_viscode_to_viscode2(df)
                df.rename(columns={'VISCODE':'VISCODE2'}, inplace=True)
                if 'EXAMDATE' in df.columns:
                    df = df.drop(['EXAMDATE'], axis=1)
                for j in range(len(df)):
                    if j == 1:
                        continue
                    '''
                    tmp1 = df.values[j][0]
                    print(tmp1)
                    if 'v' in tmp1:
                        df = df.drop(index=[j])   #delete the VISCODE which can't be transformed .
                    '''
                df = df.loc[df['VISCODE2'].str.contains('bl') | df['VISCODE2'].str.startswith('m') | df['VISCODE2'].str.contains('sc')]
                df.to_csv(Folder_Path + '/' + file_list[i])
            else :
                os.remove(Folder_Path + '/' + file_list[i])
        if 'COMMENTS' in df.columns:
            df = df.drop(['COMMENTS'], axis=1)
            df.to_csv(Folder_Path + '/' + file_list[i])
        if 'COMMENT' in df.columns:
            df = df.drop(['COMMENT'], axis=1)
            df.to_csv(Folder_Path + '/' + file_list[i])
        if 'NOTE' in df.columns:
            df = df.drop(['NOTE'], axis=1)
            df.to_csv(Folder_Path + '/' + file_list[i])

def csf_clear_data() :
    #脑脊髓液中的β样淡粉蛋白，BETA_AMYLOID_1_40：Aβ40含量， BETA_AMYLOID_1_42：Aβ42含量，BETA_AMYLOID_42_40：Aβ40/Aβ42的比值
    keep_rows(bio_path+"csf/ADNI_EUROIMMUN.csv",
              ['RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'BETA_AMYLOID_1_40', 'BETA_AMYLOID_1_42', 'BETA_AMYLOID_42_40', 'COMMENTS'],
              Csf_Path+"ADNI_EUROIMMUN.csv")

    #脑脊髓液中 sTREM2蛋白和PGRN蛋白（总数据1878条，基线数据409条，RID个数1036），该两种蛋白是否能作为AD的诊断检测生物标志物还在研究之中
    #有重复的[rid,viscode]
    keep_rows(bio_path+"csf/ADNI_HAASS_WASHU_LAB.csv",
              ['RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'WU_STREM2', 'WU_STREM2_CV', 'WU_STREM2CORRECTED', 'MSD_STREM2', 'MSD_STREM2_CV',
               'MSD_STREM2CORRECTED', 'TREM2OUTLIER', 'MSD_PGRN', 'MSD_PGRN_CV', 'MSD_PGRNCORRECTED'],
              save_path_with_duplicates+"ADNI_HAASS_WASHU_LAB_duplicates.csv")

    # region 去除重复的记录
    df_ADNI_HAASS_WASHU_LAB = pd.read_csv(bio_path+"csf/ADNI_HAASS_WASHU_LAB.csv", dtype=str)
    df_ADNI_HAASS_WASHU_LAB = df_ADNI_HAASS_WASHU_LAB.sort_values(by=['RID', 'VISCODE2','WU_STREM2'])
    df_ADNI_HAASS_WASHU_LAB=df_ADNI_HAASS_WASHU_LAB.drop_duplicates(subset=['RID', 'VISCODE2'])
    df_ADNI_HAASS_WASHU_LAB = df_ADNI_HAASS_WASHU_LAB[['RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'WU_STREM2', 'WU_STREM2_CV', 'WU_STREM2CORRECTED', 'MSD_STREM2', 'MSD_STREM2_CV',
               'MSD_STREM2CORRECTED', 'TREM2OUTLIER', 'MSD_PGRN', 'MSD_PGRN_CV', 'MSD_PGRNCORRECTED']]
    df_ADNI_HAASS_WASHU_LAB.to_csv(Csf_Path+"ADNI_HAASS_WASHU_LAB.csv", index = False)

    #脑脊髓液中的各种蛋白的含量（总数据398条，基线数据305条，RID个数386条）。这些生物标志物是否能够作为AD诊断检测还在研究之中。所以列出所有列属性：
    keep_rows(bio_path+"csf/ADNI_HULAB.csv",
              ['RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'TNFR1', 'TNFR2', 'TGFBETA1', 'TGFBETA2', 'TGFBETA3', 'IL_21', 'IL_6', 'IL_7',
               'IL_9', 'IL_10', 'IL_12_P40', 'ICAM_1', 'VCAM_1'],
              Csf_Path+"ADNI_HULAB.csv")

    #脑脊髓液数据，包含β淀粉样蛋白和tua蛋白的含量，(中数据250条，基线数据56条，RID个数250个这四种生物标志物是AD诊断检测的生物标志物)
    keep_rows(bio_path+"csf/ADNI_MESOSCALE.csv",
              ['RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'ABETA38', 'ABETA40', 'ABETA42', 'TAU'],
              Csf_Path+"ADNI_MESOSCALE.csv")

    #脊髓液数据（总数据805条，基线数据368条，RID个数372），该生物标志物是否可以作为诊断/检测的AD的生物标志物还在研究中。有用的列属性：
    ##VALUE! = BA8026H9 not measured for alpha synuclein
    ##VALUE! = ps129 below assay detection limit for DA802Z61 and HA802ZKL
    keep_rows(bio_path+"csf/ADNI_ZHANG.csv",
              ['RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'FINAL_ASYN', 'FINAL_PS129', 'FINAL_HGB', 'RUNDATE_ASYN', 'RUNDATE_HGB'],
              Csf_Path+"ADNI_ZHANG.csv")

    #脑髓液（总数据415，基线数据415，RID个数399）中神经轻链多肽的含量，现在尚不清楚该蛋白生物标志物能作为AD的生物标志物，尚在进一步的研究中。
    keep_rows(bio_path+"csf/BLENNOWCSFNFL.csv",
              ['RID', 'VISCODE', 'EXAMDATE', 'CSFNFL', 'COMMENTS'],
              save_path_with_duplicates+"BLENNOWCSFNFL_duplicates.csv")
    # region 去除重复的记录
    df_BLENNOWCSFNFL = pd.read_csv(bio_path + "csf/BLENNOWCSFNFL.csv", dtype=str)
    df_BLENNOWCSFNFL = df_BLENNOWCSFNFL.drop_duplicates(subset=['RID', 'VISCODE','EXAMDATE',])
    df_BLENNOWCSFNFL = df_BLENNOWCSFNFL[
        ['RID', 'VISCODE', 'EXAMDATE', 'CSFNFL', 'COMMENTS']]
    df_BLENNOWCSFNFL.to_csv(Csf_Path + "BLENNOWCSFNFL.csv", index=False)


    #脑脊髓液（总数据量415，基线数据415，RID数量399）中的神经粒蛋白含量，研究表现该生物标志物是和AD相关，而且是针对于MCI患者，该物质含量增加，但还处于研究中，并没直接作为AD的生物标志物。
    keep_rows(bio_path+"csf/BLENNOWCSFNG.csv",
              ['RID', 'VISCODE', 'EXAMDATE', 'CSFNG', ],
              save_path_with_duplicates+"BLENNOWCSFNG_duplicates.csv")
    # region 去除重复的记录
    df_BLENNOWCSFNG = pd.read_csv(bio_path + "csf/BLENNOWCSFNG.csv", dtype=str)
    df_BLENNOWCSFNG = df_BLENNOWCSFNG.drop_duplicates(subset=['RID', 'VISCODE', 'EXAMDATE', ])
    df_BLENNOWCSFNG = df_BLENNOWCSFNG[
        ['RID', 'VISCODE', 'EXAMDATE', 'CSFNG']]
    df_BLENNOWCSFNG.to_csv(Csf_Path + "BLENNOWCSFNG.csv", index=False)

    #脑脊髓液（总数据389条，RID个数389，没有列属性VISCODE）中的两种蛋白的含量。该蛋白是潜在的生物标志物。可用列属性：
    #ALPHA_SYN：α-突触核蛋白含量
    # HEMOGLOBIN：血红蛋白的含量
    #keep_rows(bio_path+"csf/CSFALPHASYN_03_21_14.csv",
    #          ['RID', 'EXAMDATE', 'ALPHA_SYN', 'HEMOGLOBIN', 'PROTOCOL'],
    #          Csf_Path+"CSFALPHASYN_03_21_14.csv")


    #keep_rows(bio_path+"csf/CSFC3FH.csv",
    #          ['RID', 'EXAMDATE', 'FH', 'C3'],
    #          Csf_Path+"CSFC3FH.csv")

    keep_rows(bio_path+"csf/CSFMRM.csv",
              ['RID', 'VISCODE', 'A1AT_AVLTIDEK', 'A1AT_LSITGTYDLK', 'A1AT_SVLGQLGITK', 'A1BG_NGVAQEPVHLDSPAIK', 'A1BG_SGLSTGWTQLSK',
               'A2GL_DLLLPQPDLR', 'A2GL_VAAGAFQGLR', 'A4_LVFFAEDVGSNK', 'A4_THPHFVIPYR', 'A4_WYFDVTEGK', 'AACT_ADLSGITGAR',
               'AACT_EIGELYLPK', 'AACT_NLAVSQVVHK', 'AATC_IVASTLSNPELFEEWTGNVK', 'AATC_LALGDDSPALK', 'AATC_NLDYVATSIHEAVTK',
               'AATM_FVTVQTISGTGALR', 'AFAM_DADPDTFFAK', 'AFAM_FLVNLVK', 'AFAM_LPNNVLQEK', 'ALDOA_ALQASALK', 'ALDOA_QLLLTADDR',
               'AMBP_AFIQLWAFDAVK', 'AMBP_ETLLQDFR', 'AMBP_FLYHK', 'AMD_IPVDEEAFVIDFKPR', 'AMD_IVQFSPSGK', 'AMD_NGQWTLIGR',
               'APLP2_HYQHVLAVDPEK', 'APLP2_WYFDLSK', 'APOB_IAELSATAQEIIK', 'APOB_SVSLPSLDPASAK', 'APOB_TGISPLALIK', 'APOD_VLNQELR',
               'APOE_AATVGSLAGQPLQER', 'APOE_CLAVYQAGAR', 'APOE_LAVYQAGAR', 'APOE_LGADMEDVR', 'APOE_LGPLVEQGR', 'B2MG_VEHSDLSFSK',
               'B2MG_VNHVTLSQPK', 'B3GN1_EPGEFALLR', 'B3GN1_TALASGGVLDASGDYR', 'B3GN1_YEAAVPDPR', 'BACE1_SIVDSGTTNLR',
               'BASP1_ETPAATEAPSSTPK', 'BTD_LSSGLVTAALYGR', 'BTD_SHLIIAQVAK', 'C1QB_LEQGENVFLQATDK', 'C1QB_VPGLYYFTYHASSR',
               'CA2D1_FVVTDGGITR', 'CA2D1_IKPVFIEDANFGR', 'CA2D1_TASGVNQLVDIYEK', 'CAD13_DIQGSLQDIFK', 'CAD13_INENTGSVSVTR',
               'CAD13_YEVSSPYFK', 'CADM3_EGSVPPLK', 'CADM3_GNPVPQQYLWEK', 'CADM3_SLVTVLGIPQKPIITGYK', 'CAH1_VLDALQAIK', 'CAH1_YSSLAEAASK',
               'CATA_LFAYPDTHR', 'CATD_LVDQNIFSFYLSR', 'CATD_VSTLPAITLK', 'CATD_YSQAVPAVTEGPIPEVLK', 'CATL1_VFQEPLFYEAPR',
               'CCKN_AHLGALLAR', 'CCKN_NLQNLDPSHR', 'CD14_AFPALTSLDLSDNPGLGER', 'CD14_FPAIQNLALR', 'CD14_SWLAELQQWLKPGLK',
               'CD59_AGLQVYNK', 'CERU_IYHSHIDAPK', 'CERU_NNEGTYYSPNYNPQSR', 'CFAB_DAQYAPGYDK', 'CFAB_VSEADSSNADWVTK', 'CFAB_YGLVTYATYPK',
               'CH3L1_ILGQQVPYATK', 'CH3L1_SFTLASSETGVGAPISGPGIPGR', 'CH3L1_VTIDSSYDIAK', 'CLUS_IDSLLENDR', 'CLUS_SGSGLVGR',
               'CLUS_VTTVASHTSDSDVPSGVTEVVVK', 'CMGA_EDSLEAGLPLQVR', 'CMGA_SEALAVDGAGKPGAEEAQDPEGK', 'CMGA_SGEATDGARPQALPEPMQESK',
               'CMGA_SGELEQEEER', 'CMGA_YPGPQAEGDSEGLSQGLVDR', 'CNDP1_ALEQDLPVNIK', 'CNDP1_VFQYIDLHQDEFVQTLK', 'CNDP1_WNYIEGTK',
               'CNTN1_DGEYVVEVR', 'CNTN1_TTKPYPADIVVQFK', 'CNTN2_IIVQAQPEWLK', 'CNTN2_TTGPGGDGIPAEVHIVR', 'CNTN2_VIASNILGTGEPSGPSSK',
               'CNTP2_HELQHPIIAR', 'CNTP2_VDNAPDQQNSHPDLAQEEIR', 'CNTP2_YSSSDWVTQYR', 'CO2_DFHINLFR', 'CO2_HAIILLTDGK',
               'CO2_SSGQWQTPGATR', 'CO3_IHWESASLLR', 'CO3_LSINTHPSQKPLSITVR', 'CO3_TELRPGETLNVNFLLR', 'CO3_TGLQEVEVK',
               'CO3_VPVAVQGEDTVQSLTQGDGVAK', 'CO4A_DHAVDLIQK', 'CO4A_GSFEFPVGDAVSK', 'CO4A_LGQYASPTAK', 'CO4A_NVNFQK',
               'CO4A_VLSLAQEQVGGSPEK', 'CO4A_VTASDPLDTLGSEGALSPGGVASLLR', 'CO5_DINYVNPVIK', 'CO5_TLLPVSKPEIR', 'CO5_VFQFLEK',
               'CO6_ALNHLPLEYNSALYSR', 'CO6_SEYGAALAWEK', 'CO8B_IPGIFELGISSQSDR', 'CO8B_SDLEVAHYK', 'CO8B_YEFILK', 'COCH_GVISNSGGPVR',
               'CRP_ESDTSYVSLK', 'CSTN1_GNLAGLTLR', 'CSTN1_IHGQNVPFDAVVVDK', 'CSTN1_IPDGVVSVSPK', 'CSTN3_ATGEGLIR', 'CSTN3_ESLLLDTTSLQQR',
               'CUTA_TQSSLVPALTDFVR', 'CYTC_ALDFAVGEYNK', 'DAG1_GVHYISVSATR', 'DAG1_LVPVVNNR', 'DAG1_VTIPTDLIASSGDIIK', 'DIAC_ATYIQNYR',
               'ENOG_GNPTVEVDLYTAK', 'ENOG_LGAEVYHTLK', 'ENPP2_SYPEILTLK', 'ENPP2_WWGGQPLWITATK', 'EXTL2_VIVVWNNIGEK', 'FABPH_SIVTLDGGK',
               'FABPH_SLGVGFATR', 'FAM3C_GINVALANGK', 'FAM3C_SALDTAAR', 'FAM3C_SPFEQHIK', 'FAM3C_TGEVLDTK',
               'FBLN1_AITPPHPASQANIIFDITEGNLR', 'FBLN1_IIEVEEEQEDPYLNDR', 'FBLN1_TGYYFDGISR', 'FBLN3_IPSNPSHR', 'FBLN3_LTIIVGPFSF',
               'FBLN3_SGNENGEFYLR', 'FETUA_AHYDLR', 'FETUA_FSVVYAK', 'FETUA_HTLNQIDEVK', 'FMOD_YLPFVPSR', 'GFAP_ALAAELNQLR',
               'GOLM1_DQLVIPDGQEEEQEAAGEGR', 'GOLM1_QQLQALSEPQPR', 'GRIA4_EYPGSETPPK', 'GRIA4_LQNILEQIVSVGK', 'GRIA4_NTDQEYTAFR',
               'HBA_FLASVSTVLTSK', 'HBA_TYFPHFDLSHGSAQVK', 'HBA_VGAHAGEYGAEALER', 'HBB_EFTPPVQAAYQK', 'HBB_SAVTALWGK',
               'HBB_VNVDEVGGEALGR', 'HEMO_NFPSPVDAAFR', 'HEMO_QGHNSVFLIK', 'HEMO_SGAQATWTELPWPHEK', 'I18BP_LWEGSTSR', 'IBP2_HGLYNLK',
               'IBP2_LIQGAPTIR', 'IGSF8_DTQFSYAVFK', 'IGSF8_LQGDAVVLK', 'IGSF8_VVAGEVQVQR', 'ITIH1_EVAFDLEIPK', 'ITIH1_QYYEGSEIVVAGR',
               'ITIH5_SYLEITPSR', 'KAIN_FYYLIASETPGK', 'KAIN_LGFTDLFSK', 'KAIN_VGSALFLSHNLK', 'KAIN_WADLSGITK', 'KLK10_ALQLPYR',
               'KLK11_LPHTLR', 'KLK6_ESSQEQSSVVR', 'KLK6_LSELIQPLPLER', 'KLK6_YTNWIQK', 'KNG1_DIPTNSPELEETLTHTITK', 'KNG1_QVVAGLNFR',
               'KNG1_TVGSDTFYSFK', 'KPYM_LDIDSPPITAR', 'L1CAM_AQLLVVGSPGPVPR', 'L1CAM_LVLSDLHLLTQSQVR', 'L1CAM_WRPVDLAQVK',
               'LAMB2_AQGIAQGAIR', 'LPHN1_LVVSQLNPYTLR', 'LPHN1_SGETVINTANYHDTSPYR', 'LRC4B_HLEILQLSK', 'LRC4B_LTTVPTQAFEYLSK',
               'LTBP2_EQDAPVAGLQPVER', 'MIME_ESAYLYAR', 'MIME_ETVIIPNEK', 'MIME_LEGNPIVLGK', 'MOG_VVHLYR', 'MUC18_EVTVPVFYPTEK',
               'MUC18_GATLALTQVTPQDER', 'NBL1_LALFPDK', 'NCAM1_AGEQDATIHLK', 'NCAM1_GLGEISAASEFK', 'NCAM2_ASGSPEPAISWFR',
               'NCAM2_IIELSQTTAK', 'NCAN_APVLELEK', 'NCAN_LSSAIIAAPR', 'NEGR1_SSIIFAGGDK', 'NEGR1_VVVNFAPTIQEIK', 'NEGR1_WSVDPR',
               'NELL2_AFLFQDTPR', 'NELL2_FTGSSWIK', 'NELL2_SALAYVDGK', 'NEO1_DVVASLVSTR', 'NEUS_ALGITEIFIK', 'NEUS_QEVPLATLEPLVK',
               'NGF_SAPAAAIAAR', 'NICA_ALADVATVLGR', 'NICA_APDVTTLPR', 'NPTX1_FQLTFPLR', 'NPTX1_LENLEQYSR', 'NPTX2_LESLEHQLR',
               'NPTX2_TESTLNALLQR', 'NPTXR_ELDVLQGR', 'NPTXR_LVEAFGGATK', 'NRCAM_SLPSEASEQYLTK', 'NRCAM_VFNTPEGVPSAPSSLK',
               'NRCAM_YIVSGTPTFVPYLIK', 'NRX1A_DLFIDGQSK', 'NRX1A_ITTQITAGAR', 'NRX1A_SDLYIGGVAK', 'NRX2A_AIVADPVTFK',
               'NRX2A_LGERPPALLGSQGLR', 'NRX2A_LSALTLSTVK', 'NRX3A_IYGEVVFK', 'NRX3A_SDLSFQFK', 'OSTP_AIPVAQDLNAPSDWDSR',
               'PCSK1_ALAHLLEAER', 'PCSK1_GEAAGAVQELAR', 'PCSK1_NSDPALGLDDDPDAPAAQLAR', 'PDYN_FLPSISTK', 'PDYN_LSGSFLK',
               'PDYN_SVGEGPYSELAK', 'PEDF_DTDTGALLFIGK', 'PEDF_SSFVAPLEK', 'PEDF_TVQAVLTVPK', 'PGRP2_AGLLRPDYALLGHR', 'PGRP2_TFTLLDPK',
               'PIMT_VQLVVGDGR', 'PLDX1_LYGPSEPHSR', 'PLMN_EAQLPVIENK', 'PLMN_HSIFTPETNPR', 'PLMN_LSSPAVITDK', 'PPN_VHQSPDGTLLIYNLR',
               'PRDX1_DISLSDYK', 'PRDX1_LVQAFQFTDK', 'PRDX2_GLFIIDGK', 'PRDX2_IGKPAPDFK', 'PRDX3_HLSVNDLPVGR', 'PRDX6_LIALSIDSVEDHLAWSK',
               'PRDX6_LSILYPATTGR', 'PTGDS_AQGFTEDTIVFLPQTDK', 'PTGDS_WFSAGLASNSSWLR', 'PTPRN_AEAPALFSR', 'PTPRN_LAAVLAGYGVELR',
               'PTPRN_SELEAQTGLQILQTGVGQR', 'PVRL1_ITQVTWQK', 'SCG1_GEAGAPGEEDIQGPTK', 'SCG1_HLEEPGETQNAFLNER', 'SCG1_NYLNYGEEGAPGK',
               'SCG1_SSQGGSLPSEEK', 'SCG2_ALEYIENLR', 'SCG2_IILEALR', 'SCG2_VLEYLNQEK', 'SCG3_ELSAERPLNEQIAEAEEDK',
               'SCG3_FQDDPDGLHQLDGTPLTAEDIVHK', 'SCG3_LNVEDVDSTK', 'SE6L1_ETGTPIWTSR', 'SE6L1_SPTNTISVYFR', 'SIAE_ELSNTAAYQSVR',
               'SLIK1_SLPVDVFAGVSLSK', 'SODC_GDGPVQGIINFEQK', 'SODC_HVGDLGNVTADK', 'SODC_TLVVHEK', 'SODE_AGLAASLAGPHSIVGR',
               'SODE_AVVVHAGEDDLGR', 'SODE_VTGVVLFR', 'SORC1_TIAVYEEFR', 'SPON1_VTLSAAPPSYFR', 'SPRL1_HIQETEWQSQEGK',
               'SPRL1_HSASDDYFIPSQAFLEAER', 'SPRL1_VLTHSELAPLR', 'TGFB1_LLAPSDSPEWLSFDVTGVVR', 'THRB_ETAASLLQAGYK',
               'THRB_YGFYTHVFR', 'TIMP1_GFQALGDAADIR', 'TIMP1_SEEFLIAGK', 'TNR21_ASNLIGTYR', 'TRFM_ADTDGGLIFR',
               'TTHY_TSESGELHGLTTEEEFVEGIYK', 'TTHY_VEIDTK', 'UBB_ESTLHLVLR', 'UBB_TITLEVEPSDTIENVK', 'UBB_TLSDYNIQK',
               'VASN_NLHDLDVSDNQLER', 'VASN_SLTLGIEPVSPTSLR', 'VASN_YLQGSSVQLR', 'VGF_AYQGVAAPFPK', 'VGF_NSEPQDEGELFQGVDPR',
               'VGF_THLGEALAPLSK', 'VTDB_EFSHLGK', 'VTDB_HLSLLTTLSNR', 'VTDB_VPTADLEDVLPLAEDITNILSK'],
              Csf_Path+"CSFMRM.csv")     #with lower prio because the protein is unkonow

    #脊髓液（总数据587，基线数据147，RID个数152）中的生物标志物，其中包括粘蛋白样蛋白-1（Visinin-like protein-1,）、体相关蛋白25（synaptosomal-associated protein-25）、几丁质酶-3样蛋白-1（chitinase-3-like-1）、突触神经粒蛋白（neurogranin）是CFS中新兴的标志物。可以查看文献：https://www.sciencedirect.com/science/article/abs/pii/S1552526019300135 。
    keep_rows(bio_path+"csf/FAGANLAB_07_15_2015.csv",
              ['RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'VILIP', 'VILIP_STDEV', 'VILIP_CV', 'YKL', 'YKL_STDEV', 'YKL_CV', 'SNAP', 'SNAP_STDEV', 'SNAP_CV', 'NGRN', 'NGRN_STDEV', 'NGRN_CV'],
              Csf_Path+"FAGANLAB_07_15_2015.csv")

    #脑脊髓液（总数据5911条，基线数据972，RID个数1896）数据，其中有用的数据列：
    keep_rows(bio_path+"csf/LOCLAB.csv",
              ['RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'CTWHITE', 'CTRED', 'PROTEIN', 'GLUCOSE'],
              Csf_Path+"LOCLAB.csv")

    #脑脊髓液（共2401条数据，基线数据513条，RID个数1275）数据，其中列表数据某种生物标志物的含量超过限定值，会在列属性 comments中说明。有用的列属性包括
    keep_rows(bio_path+"csf/UPENNBIOMK9_04_19_17.csv",
              ['RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'ABETA', 'TAU', 'PTAU'],   #COMMENT
              Csf_Path+"UPENNBIOMK9_04_19_17.csv")

    #脊髓液（一共442条数据，基线数据31个 ，RID个数184）信息
    keep_rows(bio_path+"csf/UPENNBIOMKADNIDIAN2017.csv",
              ['RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'ABETA', 'AB40', 'TAU', 'PTAU', 'A4240'],    #NOTE
              Csf_Path+"UPENNBIOMKADNIDIAN2017.csv")

    #脑脊髓液（5876条数据，基线数据5742条，RID个数=1249）数据（其中数据都是经过re-scaling-缩放处理后的结果，对实际测得的数据进行了缩放），可用的列属性：
    #有重复的[rid,viscode]
    keep_rows(bio_path+"csf/UPENNBIOMK_MASTER.csv",
              ['RID', 'VISCODE', 'ABETA', 'TAU', 'PTAU', 'ABETA_RAW', 'TAU_RAW', 'PTAU_RAW'],
              save_path_with_duplicates+"UPENNBIOMK_MASTER_duplicates.csv")

    #   keep_rows("../tmp_table/csf/", , "1_tmp/") UPENNBIOMK_MASTER.csv
    #region 去除重复的记录 BATCH
    df_UPENNBIOMK_MASTER=pd.read_csv(bio_path+"csf/UPENNBIOMK_MASTER.csv", dtype=str)
    df_UPENNBIOMK_MASTER=df_UPENNBIOMK_MASTER.sort_values(by=['RID', 'VISCODE','BATCH'])

    index_list=[]
    c_rid='-1'
    c_viscode='-1'
    tmp_BATCH=[]
    for order_index, row in df_UPENNBIOMK_MASTER.iterrows():
        rid=row['RID']
        viscode=row['VISCODE']

        if rid==c_rid and viscode==c_viscode:
            tmp_BATCH.append([order_index,row['BATCH']])
        else:
            if len(tmp_BATCH)>0:
                if len(tmp_BATCH)==1:
                    index_list.append(tmp_BATCH[0][0])
                elif len(tmp_BATCH)==2:
                    if tmp_BATCH[0][1]=='MEDIAN' or tmp_BATCH[0][1]=='UPENNBIOMK':
                        index_list.append(tmp_BATCH[1][0])
                    else:
                        index_list.append(tmp_BATCH[0][0])
                else:
                    if (tmp_BATCH[0][1] == 'MEDIAN' or tmp_BATCH[0][1] == 'UPENNBIOMK') and tmp_BATCH[1][1] == 'UPENNBIOMK':
                        index_list.append(tmp_BATCH[2][0])
                    elif (tmp_BATCH[0][1] == 'MEDIAN' or tmp_BATCH[0][1] == 'UPENNBIOMK') and tmp_BATCH[1][1] != 'UPENNBIOMK':
                        index_list.append(tmp_BATCH[1][0])
                    else:
                        index_list.append(tmp_BATCH[0][0])
            tmp_BATCH.clear()
            c_rid=rid
            c_viscode=viscode
            tmp_BATCH.append([order_index, row['BATCH']])
    if len(tmp_BATCH) > 0:
        if len(tmp_BATCH) == 1:
            index_list.append(tmp_BATCH[0][0])
        elif len(tmp_BATCH) == 2:
            if tmp_BATCH[0][1] == 'MEDIAN' or tmp_BATCH[0][1] == 'UPENNBIOMK':
                index_list.append(tmp_BATCH[1][0])
            else:
                index_list.append(tmp_BATCH[0][0])
        else:
            if (tmp_BATCH[0][1] == 'MEDIAN' or tmp_BATCH[0][1] == 'UPENNBIOMK') and tmp_BATCH[1][1] == 'UPENNBIOMK':
                index_list.append(tmp_BATCH[2][0])
            elif (tmp_BATCH[0][1] == 'MEDIAN' or tmp_BATCH[0][1] == 'UPENNBIOMK') and tmp_BATCH[1][1] != 'UPENNBIOMK':
                index_list.append(tmp_BATCH[1][0])
            else:
                index_list.append(tmp_BATCH[0][0])

    df_UPENNBIOMK_MASTER=df_UPENNBIOMK_MASTER.loc[index_list,['RID', 'VISCODE', 'ABETA', 'TAU', 'PTAU', 'ABETA_RAW', 'TAU_RAW', 'PTAU_RAW']]
    df_UPENNBIOMK_MASTER.rename(columns={'TAU': 'TAU_4', 'ABETA': 'ABETA_4', 'PTAU': 'PTAU_4'}, inplace=True)
    df_UPENNBIOMK_MASTER.to_csv(Csf_Path+"UPENNBIOMK_MASTER.csv", index=False)
    #endregion



def urine_clear_data() :
    keep_rows(bio_path+"urine/ISOPROSTANE.csv", ['RID', 'VISCODE2', 'ISO8PGF2A', 'ISO812IPF2A'], Urine_Path+"ISOPROSTANE.csv")

def plasma_clear_data() :
    #血浆数据（总数据218条，基线数据218，RID个数218）中的二氯二苯二氯乙烯（dichlorodiphenyldichloroethylene）的含量，该生物标志物还在研究中。
    keep_rows(bio_path+"plasma/ADRC_EMORY_DDE.csv", ['RID', 'VISCODE2', 'ANALYSIS_BATCH', 'DDE_RECOVER_13C_PERCENT', 'DDE_PLASMA_CONCENTRATION', 'DETECTION_LIMIT'], Plasma_Path+"ADRC_EMORY_DDE.csv")

    #HCRES，血浆（总数据3390，基线数据815，RID个数819）中的高同型半胱氨酸信息，而且处以研究阶段，多个文章阐述此生物标志物可能与AD相关。在中国痴呆诊疗指南P22页，表格中的排除标准。其中有用的列属性：
    keep_rows(bio_path+"plasma/HCRES.csv", ['RID', 'VISCODE2', 'HCAMPLAS', 'HCVOLUME', 'HCRECEIVE', 'HCFROZEN', 'HCRESAMP', 'HCUSABLE'], Plasma_Path+"HCRES.csv")

    #血浆信息（总数据3762条，基线数据122，RID个数1191），神经丝轻型多肽蛋白（neurofilament light）含量，该蛋白是否能作为AD的诊断检测生物标志物还在研究之中。给出该数据有用的列属性
    #keep_rows(bio_path+"plasma/ADNI_BLENNOWPLASMANFL_10_03_18.csv", ['RID', 'VISCODE2', 'PLASMA_NFL'], Plasma_Path+"ADNI_BLENNOWPLASMANFL_10_03_18.csv")

    #血浆信息（总数据3762条，基线数据122，RID个数1191），神经丝轻型多肽蛋白（neurofilament light）含量，该蛋白是否能作为AD的诊断检测生物标志物还在研究之中。给出该数据有用的列属性：
    #keep_rows(bio_path+"plasma/ADNI_BLENNOWPLASMANFLLONG_10_03_18.csv", ['RID', 'VISCODE2', 'PLASMA_NFL'], Plasma_Path+"ADNI_BLENNOWPLASMANFLLONG_10_03_18.csv")

    #合并信息
    df_bl1 = pd.read_csv(bio_path+"plasma/ADNI_BLENNOWPLASMANFL_10_03_18.csv", dtype=str)
    df_bl1=df_bl1[['RID', 'VISCODE2', 'PLASMA_NFL']]
    df_bl2= pd.read_csv(bio_path+"plasma/ADNI_BLENNOWPLASMANFLLONG_10_03_18.csv", dtype=str)
    df_bl2=df_bl2[['RID', 'VISCODE2', 'PLASMA_NFL']]
    df_bl=pd.concat([df_bl1,df_bl2],axis=0,sort=False)
    path = os.path.dirname(Plasma_Path+"ADNI_BLENNOWPLASMANFL_BLENNOWPLASMANFLLONG.csv")
    if not os.path.exists(path):
        os.makedirs(path)
    df_bl = df_bl.drop_duplicates()
    df_bl.to_csv(Plasma_Path+"ADNI_BLENNOWPLASMANFL_BLENNOWPLASMANFLLONG.csv", index = False)




    ##用LC-MS分析血浆（（总数居622条，基线数据90条，RID个数200））中的β淀粉样蛋白，该是AD的诊断生物标志物。可用的列属性：
    #keep_rows(bio_path+"plasma/batemanlab_20190621.csv", ['RID', 'VISCODE2', 'PEAK_AREA_ABETA38_N14_B07',	'PEAK_AREA_ABETA38_N14_B08',	'PEAK_AREA_ABETA38_N14_B09',	'PEAK_AREA_ABETA38_N15_B07',	'PEAK_AREA_ABETA38_N15_B08',	'PEAK_AREA_ABETA38_N15_B09',	'PEAK_AREA_ABETA40_N14_B05',	'PEAK_AREA_ABETA40_N14_B07',	'PEAK_AREA_ABETA40_N14_B08',	'PEAK_AREA_ABETA40_N14_B09',	'PEAK_AREA_ABETA40_N14_B10',	'PEAK_AREA_ABETA40_N14_B11',	'PEAK_AREA_ABETA40_N15_B05',	'PEAK_AREA_ABETA40_N15_B07',	'PEAK_AREA_ABETA40_N15_B08',	'PEAK_AREA_ABETA40_N15_B09',	'PEAK_AREA_ABETA40_N15_B10',	'PEAK_AREA_ABETA40_N15_B11',	'PEAK_AREA_ABETA42_N14_B08',	'PEAK_AREA_ABETA42_N14_B09',	'PEAK_AREA_ABETA42_N14_B10',	'PEAK_AREA_ABETA42_N14_B11',	'PEAK_AREA_ABETA42_N14_B12',	'PEAK_AREA_ABETA42_N14_B13',	'PEAK_AREA_ABETA42_N15_B08',	'PEAK_AREA_ABETA42_N15_B09',	'PEAK_AREA_ABETA42_N15_B10',	'PEAK_AREA_ABETA42_N15_B11',	'PEAK_AREA_ABETA42_N15_B12',	'PEAK_AREA_ABETA42_N15_B13',	'PEAK_AREA_ABETAMD_N14_B04',	'PEAK_AREA_ABETAMD_N14_B05',	'PEAK_AREA_ABETAMD_N14_B06',	'PEAK_AREA_ABETAMD_N14_B07',	'PEAK_AREA_ABETAMD_N14_B08',	'PEAK_AREA_ABETAMD_N14_B09',	'PEAK_AREA_ABETAMD_N14_B10',	'PEAK_AREA_ABETAMD_N14_B11',	'PEAK_AREA_ABETAMD_N15_B04',	'PEAK_AREA_ABETAMD_N15_B05',	'PEAK_AREA_ABETAMD_N15_B06',	'PEAK_AREA_ABETAMD_N15_B07',	'PEAK_AREA_ABETAMD_N15_B08',	'PEAK_AREA_ABETAMD_N15_B09',	'PEAK_AREA_ABETAMD_N15_B10',	'PEAK_AREA_ABETAMD_N15_B11',	'PEAK_AREA_ABETA40_N14_TOUSE',	'PEAK_AREA_ABETA40_N15_TOUSE',	'PEAK_AREA_ABETA42_N14_TOUSE',	'PEAK_AREA_ABETA42_N15_TOUSE',	'RATIO_ABETA42_40_BY_ISTD_TOUSE'], Plasma_Path+"batemanlab_20190621.csv")


    #PLASMAABETA.csv，血浆数据（总数据784条，基线数据300条，RID个数=305），对于数据中的非定量数据说明：ND:不可检测; >ULOQ:超过定量上限。可用具体列属性：
    keep_rows(bio_path+"plasma/PLASMAABETA.csv", ['RID', 'VISCODE2', 'FP40', 'TP40', 'FP42', 'TP42'], Plasma_Path+"PLASMAABETA.csv")

    #血浆中的各种物质的浓度（总数据1697，基线数据925，RID个数1646），这些生物标志物是否能够作为AD诊断检测还在研究之中。给出一部分列属性：
    #NDEF
    keep_rows(bio_path+"plasma/ADNINIGHTINGALE_20180827.csv", ['RID', 'VISCODE2', 'XXL_VLDL_P',	'XXL_VLDL_L',	'XXL_VLDL_PL',	'XXL_VLDL_C',	'XXL_VLDL_CE',	'XXL_VLDL_FC',	'XXL_VLDL_TG',	'XL_VLDL_P',	'XL_VLDL_L',	'XL_VLDL_PL',	'XL_VLDL_C',	'XL_VLDL_CE',	'XL_VLDL_FC',	'XL_VLDL_TG',	'L_VLDL_P',	'L_VLDL_L',	'L_VLDL_PL',	'L_VLDL_C',	'L_VLDL_CE',	'L_VLDL_FC',	'L_VLDL_TG',	'M_VLDL_P',	'M_VLDL_L',	'M_VLDL_PL',	'M_VLDL_C',	'M_VLDL_CE',	'M_VLDL_FC',	'M_VLDL_TG',	'S_VLDL_P',	'S_VLDL_L',	'S_VLDL_PL',	'S_VLDL_C',	'S_VLDL_CE',	'S_VLDL_FC',	'S_VLDL_TG',	'XS_VLDL_P',	'XS_VLDL_L',	'XS_VLDL_PL',	'XS_VLDL_C',	'XS_VLDL_CE',	'XS_VLDL_FC',	'XS_VLDL_TG',	'IDL_P',	'IDL_L',	'IDL_PL',	'IDL_C',	'IDL_CE',	'IDL_FC',	'IDL_TG',	'L_LDL_P',	'L_LDL_L',	'L_LDL_PL',	'L_LDL_C',	'L_LDL_CE',	'L_LDL_FC',	'L_LDL_TG',	'M_LDL_P',	'M_LDL_L',	'M_LDL_PL',	'M_LDL_C',	'M_LDL_CE',	'M_LDL_FC',	'M_LDL_TG',	'S_LDL_P',	'S_LDL_L',	'S_LDL_PL',	'S_LDL_C',	'S_LDL_CE',	'S_LDL_FC',	'S_LDL_TG',	'XL_HDL_P',	'XL_HDL_L',	'XL_HDL_PL',	'XL_HDL_C',	'XL_HDL_CE',	'XL_HDL_FC',	'XL_HDL_TG',	'L_HDL_P',	'L_HDL_L',	'L_HDL_PL',	'L_HDL_C',	'L_HDL_CE',	'L_HDL_FC',	'L_HDL_TG',	'M_HDL_P',	'M_HDL_L',	'M_HDL_PL',	'M_HDL_C',	'M_HDL_CE',	'M_HDL_FC',	'M_HDL_TG',	'S_HDL_P',	'S_HDL_L',	'S_HDL_PL',	'S_HDL_C',	'S_HDL_CE',	'S_HDL_FC',	'S_HDL_TG',	'XXL_VLDL_PL__',	'XXL_VLDL_C__',	'XXL_VLDL_CE__',	'XXL_VLDL_FC__',	'XXL_VLDL_TG__',	'XL_VLDL_PL__',	'XL_VLDL_C__',	'XL_VLDL_CE__',	'XL_VLDL_FC__',	'XL_VLDL_TG__',	'L_VLDL_PL__',	'L_VLDL_C__',	'L_VLDL_CE__',	'L_VLDL_FC__',	'L_VLDL_TG__',	'M_VLDL_PL__',	'M_VLDL_C__',	'M_VLDL_CE__',	'M_VLDL_FC__',	'M_VLDL_TG__',	'S_VLDL_PL__',	'S_VLDL_C__',	'S_VLDL_CE__',	'S_VLDL_FC__',	'S_VLDL_TG__',	'XS_VLDL_PL__',	'XS_VLDL_C__',	'XS_VLDL_CE__',	'XS_VLDL_FC__',	'XS_VLDL_TG__',	'IDL_PL__',	'IDL_C__',	'IDL_CE__',	'IDL_FC__',	'IDL_TG__',	'L_LDL_PL__',	'L_LDL_C__',	'L_LDL_CE__',	'L_LDL_FC__',	'L_LDL_TG__',	'M_LDL_PL__',	'M_LDL_C__',	'M_LDL_CE__',	'M_LDL_FC__',	'M_LDL_TG__',	'S_LDL_PL__',	'S_LDL_C__',	'S_LDL_CE__',	'S_LDL_FC__',	'S_LDL_TG__',	'XL_HDL_PL__',	'XL_HDL_C__',	'XL_HDL_CE__',	'XL_HDL_FC__',	'XL_HDL_TG__',	'L_HDL_PL__',	'L_HDL_C__',	'L_HDL_CE__',	'L_HDL_FC__',	'L_HDL_TG__',	'M_HDL_PL__',	'M_HDL_C__',	'M_HDL_CE__',	'M_HDL_FC__',	'M_HDL_TG__',	'S_HDL_PL__',	'S_HDL_C__',	'S_HDL_CE__',	'S_HDL_FC__',	'S_HDL_TG__',	'VLDL_D',	'LDL_D',	'HDL_D',	'SERUM_C',	'VLDL_C',	'REMNANT_C',	'LDL_C',	'HDL_C',	'HDL2_C',	'HDL3_C',	'ESTC',	'FREEC',	'SERUM_TG',	'VLDL_TG',	'LDL_TG',	'HDL_TG',	'TOTPG',	'TG_PG',	'PC',	'SM',	'TOTCHO',	'APOA1',	'APOB',	'APOB_APOA1',	'TOTFA',	'UNSAT',	'DHA',	'LA',	'FAW3',	'FAW6',	'PUFA',	'MUFA',	'SFA',	'DHA_FA',	'LA_FA',	'FAW3_FA',	'FAW6_FA',	'PUFA_FA',	'MUFA_FA',	'SFA_FA',	'GLC',	'LAC',	'PYR',	'CIT',	'GLOL',	'ALA',	'GLN',	'GLY',	'HIS',	'ILE',	'LEU',	'VAL',	'PHE',	'TYR',	'ACE',	'ACACE',	'BOHBUT',	'CREA',	'ALB',	'GP'],Plasma_Path+"ADNINIGHTINGALE_20180827.csv")

    #血浆（总数据量581，基线数据581，RID数量581）中的tau蛋白含量，该蛋白是AD的生物标志物。可用列属性：<lloq,<LOD
    keep_rows(bio_path+"plasma/BLENNOWPLASMATAU.csv", ['RID', 'VISCODE2', 'PLASMATAU'], Plasma_Path+"BLENNOWPLASMATAU.csv")

    #血浆数据
    keep_rows(bio_path+"plasma/UPENNPLASMA.csv", ['RID', 'VISCODE2', 'AB40', 'AB42'], Plasma_Path+"UPENNPLASMA.csv")



    #以下3个由血清合并上来
    #血清数据（总数据829条，基线数据941条，RID个数875），测定血清中酰基肉碱、氨基酸、生物胺、甘油磷脂和鞘磷脂等，这些生物标志物是否能够作为AD诊断检测还在研究之中。因为其说明没有明确的说明，在此处就不在给出相关列属性的说明。
    keep_rows(bio_path + "serum/ADMCDUKEP180UPLCADNI2GO.csv",['RID', 'VISCODE', 'Ala',	'Arg',	'Asn',	'Asp',	'Cit',	'Gln',	'Glu',	'Gly',	'His',	'Ile',	'Lys',	'Met',	'Orn',	'Phe',	'Pro',	'Ser',	'Thr',	'Trp',	'Tyr',	'Val',	'Ac.Orn',	'ADMA',	'alpha.AAA',	'c4.OH.Pro',	'canosine',	'Creatinine',	'DOPA',	'Dopamine',	'Histamine',	'Kynurenine',	'Met.So',	'Nitro.Tyr',	'PEA',	'Putrescine',	'Sarcosine',	'Serotonin',	'Spermidine',	'Spermine',	't4.OH.Pro',	'Taurine',	'SDMA'],Plasma_Path + "ADMCDUKEP180UPLCADNI2GO.csv")

    #ADMC_BA_POSTPROC_06_28_18.csv，血清数据（总数据1671条，基线数据1671条，RID个数1671），酸类代谢物含量。该生物标志物是否能够作为AD诊断检测还在研究之中。给出可用的列属性：
    keep_rows(bio_path + "serum/ADMC_BA_POSTPROC_06_28_18.csv", ['RID', 'VISCODE',	'CA',	'CDCA',	'DCA',	'GCA',	'GCDCA',	'GDCA',	'GLCA',	'GUDCA',	'TCA',	'TCDCA',	'TDCA',	'TLCA',	'TMCA_A_B',	'TUDCA',	'UDCA',	'CA_CDCA',	'DCA_CA',	'GLCA_CDCA',	'GDCA_CA',	'GDCA_DCA',	'TDCA_CA',	'TLCA_CDCA',	'TDCA_DCA',	'CA_LOGTRANSFORMFLAG',	'CDCA_LOGTRANSFORMFLAG',	'DCA_LOGTRANSFORMFLAG',	'GCA_LOGTRANSFORMFLAG',	'GCDCA_LOGTRANSFORMFLAG',	'GDCA_LOGTRANSFORMFLAG',	'GLCA_LOGTRANSFORMFLAG',	'GUDCA_LOGTRANSFORMFLAG',	'TCA_LOGTRANSFORMFLAG',	'TCDCA_LOGTRANSFORMFLAG',	'TDCA_LOGTRANSFORMFLAG',	'TLCA_LOGTRANSFORMFLAG', 'TMCA_A_B_LOGTRANSFORMFLAG',	'TUDCA_LOGTRANSFORMFLAG',	'UDCA_LOGTRANSFORMFLAG',	'CA_CDCA_LOGTRANSFORMFLAG',	'DCA_CA_LOGTRANSFORMFLAG',	'GLCA_CDCA_LOGTRANSFORMFLAG',	'GDCA_CA_LOGTRANSFORMFLAG',	'GDCA_DCA_LOGTRANSFORMFLAG',	'TDCA_CA_LOGTRANSFORMFLAG',	'TLCA_CDCA_LOGTRANSFORMFLAG',	'TDCA_DCA_LOGTRANSFORMFLAG'],Plasma_Path + "ADMC_BA_POSTPROC_06_28_18.csv")

    #血清数据（总数据732条，基线数据941条，RID个数875），用流动注射分析法FIA-MS/MS测得血清中代谢物:酰基肉碱、氨基酸、生物胺、甘油磷脂和鞘磷脂等。这些生物标志物是否能够作为AD诊断检测还在研究之中。因为其说明没有明确的说明，在此处就不在给出相关列属性的说明。
    keep_rows(bio_path + "serum/ADMCDUKEP180FIAADNI2GO.csv", ['RID', 'VISCODE',	'C0',	'C10',	'C10.1',	'C10.2',	'C12',	'C12.DC',	'C12.1',	'C14',	'C14.1',	'C14.1.OH',	'C14.2',	'C14.2.OH',	'C16',	'C16.OH',	'C16.1',	'C16.1.OH',	'C16.2',	'C16.2.OH',	'C18',	'C18.1',	'C18.1.OH',	'C18.2',	'C2',	'C3',	'C3.DC..C4.OH.',	'C3.OH',	'C3.1',	'C4',	'C4.1',	'C6..C4.1.DC.',	'C5',	'C5.M.DC',	'C5.OH..C3.DC.M.',	'C5.1',	'C5.1.DC',	'C5.DC..C6.OH.',	'C6.1',	'C7.DC',	'C8',	'C9',	'lysoPC.a.C14.0',	'lysoPC.a.C16.0',	'lysoPC.a.C16.1',	'lysoPC.a.C17.0',	'lysoPC.a.C18.0',	'lysoPC.a.C18.1',	'lysoPC.a.C18.2',	'lysoPC.a.C20.3',	'lysoPC.a.C20.4',	'lysoPC.a.C24.0',	'lysoPC.a.C26.0',	'lysoPC.a.C26.1',	'lysoPC.a.C28.0',	'lysoPC.a.C28.1',	'PC.aa.C24.0',	'PC.aa.C26.0',	'PC.aa.C28.1',	'PC.aa.C30.0',	'PC.aa.C32.0',	'PC.aa.C32.1',	'PC.aa.C32.3',	'PC.aa.C34.1',	'PC.aa.C34.2',	'PC.aa.C34.3',	'PC.aa.C34.4',	'PC.aa.C36.0',	'PC.aa.C36.1',	'PC.aa.C36.2',	'PC.aa.C36.3',	'PC.aa.C36.4',	'PC.aa.C36.5',	'PC.aa.C36.6',	'PC.aa.C38.0',	'PC.aa.C38.3',	'PC.aa.C38.4',	'PC.aa.C38.5',	'PC.aa.C38.6',	'PC.aa.C40.1',	'PC.aa.C40.2',	'PC.aa.C40.3',	'PC.aa.C40.4',	'PC.aa.C40.5',	'PC.aa.C40.6',	'PC.aa.C42.0',	'PC.aa.C42.1',	'PC.aa.C42.2',	'PC.aa.C42.4',	'PC.aa.C42.5',	'PC.aa.C42.6',	'PC.ae.C30.0',	'PC.ae.C30.1',	'PC.ae.C30.2',	'PC.ae.C32.1',	'PC.ae.C32.2',	'PC.ae.C34.0',	'PC.ae.C34.1',	'PC.ae.C34.2',	'PC.ae.C34.3',	'PC.ae.C36.0',	'PC.ae.C36.1',	'PC.ae.C36.2',	'PC.ae.C36.3',	'PC.ae.C36.4',	'PC.ae.C36.5',	'PC.ae.C38.0',	'PC.ae.C38.1',	'PC.ae.C38.2',	'PC.ae.C38.3',	'PC.ae.C38.4',	'PC.ae.C38.5',	'PC.ae.C38.6',	'PC.ae.C40.1',	'PC.ae.C40.2',	'PC.ae.C40.3',	'PC.ae.C40.4',	'PC.ae.C40.5',	'PC.ae.C40.6',	'PC.ae.C42.0',	'PC.ae.C42.1',	'PC.ae.C42.2',	'PC.ae.C42.3',	'PC.ae.C42.4',	'PC.ae.C42.5',	'PC.ae.C44.3',	'PC.ae.C44.4',	'PC.ae.C44.5',	'PC.ae.C44.6',	'SM..OH..C14.1',	'SM..OH..C16.1',	'SM..OH..C22.1',	'SM..OH..C22.2',	'SM..OH..C24.1',	'SM.C16.0',	'SM.C16.1',	'SM.C18.0',	'SM.C18.1',	'SM.C20.2',	'SM.C24.0',	'SM.C24.1',	'SM.C26.0',	'SM.C26.1'],Plasma_Path + "ADMCDUKEP180FIAADNI2GO.csv")

def serum_clear_data() :
    keep_rows(bio_path+"serum/ADMCDUKEP180UPLCADNI2GO.csv", ['RID', 'VISCODE', 'Serotonin', 'Sarcosine', 'Kynurenine', 'Taurine'], Serum_Path+"ADMCDUKEP180UPLCADNI2GO.csv")
    keep_rows(bio_path+"serum/ADMC_BA_POSTPROC_06_28_18.csv", ['RID', 'VISCODE', 'CA', 'GCA', 'GDCA'], Serum_Path+"ADMC_BA_POSTPROC_06_28_18.csv")
    keep_rows(bio_path+"serum/ADMCDUKEP180FIAADNI2GO.csv", ['RID', 'VISCODE', 'C0', 'C2', 'C3', 'C10', 'C18'], Serum_Path+"ADMCDUKEP180FIAADNI2GO.csv")

def csv_merge(type, Folder_Path, SaveFile_Path, SaveFile_Name) :
    file_list = os.listdir(Folder_Path)
    df1 = pd.read_csv(Folder_Path + file_list[0], dtype=str)

    if type == "csf" or type == "plasma" or type == "serum":
        for i in range(1, len(file_list)):
            df2 = pd.read_csv(Folder_Path + file_list[i], dtype=str)
            df1 = pd.merge(df1, df2, on=['RID', 'VISCODE2'], how='outer')
        # print(df1)

    df1.dropna(axis=0, how='all')    #delete the row which are all null value
    df1.dropna(axis=1, how='all')    #delete the line which are all null value


    df3 = df1.loc[df1['VISCODE2'].str.contains('bl') | df1['VISCODE2'].str.startswith('m') | df1['VISCODE2'].str.contains('sc') ] #delete the rows containing wrong viscode2

    if type == "csf" :
        df3 = df3[['RID', 'VISCODE2', 'BETA_AMYLOID_1_40', 'BETA_AMYLOID_1_42', 'BETA_AMYLOID_42_40','WU_STREM2', 'WU_STREM2_CV', 'WU_STREM2CORRECTED', 'MSD_STREM2',
                   'MSD_STREM2_CV','MSD_STREM2CORRECTED', 'MSD_PGRN', 'MSD_PGRN_CV', 'MSD_PGRNCORRECTED', 'TNFR1', 'TNFR2', 'TGFBETA1', 'TGFBETA2', 'TGFBETA3', 'IL_21',
                   'IL_6', 'IL_7','IL_9', 'IL_10', 'IL_12_P40', 'ICAM_1', 'VCAM_1', 'ABETA38', 'ABETA40', 'ABETA42', 'TAU_1', 'FINAL_ASYN', 'FINAL_PS129', 'FINAL_HGB',
                   'CSFNFL', 'CSFNG', 'VILIP', 'VILIP_STDEV', 'VILIP_CV', 'YKL', 'YKL_STDEV', 'YKL_CV', 'SNAP', 'SNAP_STDEV', 'SNAP_CV', 'NGRN', 'NGRN_STDEV', 'NGRN_CV',
                   'CTWHITE', 'CTRED', 'PROTEIN', 'GLUCOSE','ABETA_2', 'TAU_2', 'PTAU_2', 'ABETA_3', 'AB40', 'TAU_3', 'PTAU_3', 'A4240', 'ABETA_4', 'TAU_4', 'PTAU_4',
                   'ABETA_RAW', 'TAU_RAW', 'PTAU_RAW', 'A1AT_AVLTIDEK', 'A1AT_LSITGTYDLK', 'A1AT_SVLGQLGITK', 'A1BG_NGVAQEPVHLDSPAIK', 'A1BG_SGLSTGWTQLSK',
               'A2GL_DLLLPQPDLR', 'A2GL_VAAGAFQGLR', 'A4_LVFFAEDVGSNK', 'A4_THPHFVIPYR', 'A4_WYFDVTEGK', 'AACT_ADLSGITGAR',
               'AACT_EIGELYLPK', 'AACT_NLAVSQVVHK', 'AATC_IVASTLSNPELFEEWTGNVK', 'AATC_LALGDDSPALK', 'AATC_NLDYVATSIHEAVTK',
               'AATM_FVTVQTISGTGALR', 'AFAM_DADPDTFFAK', 'AFAM_FLVNLVK', 'AFAM_LPNNVLQEK', 'ALDOA_ALQASALK', 'ALDOA_QLLLTADDR',
               'AMBP_AFIQLWAFDAVK', 'AMBP_ETLLQDFR', 'AMBP_FLYHK', 'AMD_IPVDEEAFVIDFKPR', 'AMD_IVQFSPSGK', 'AMD_NGQWTLIGR',
               'APLP2_HYQHVLAVDPEK', 'APLP2_WYFDLSK', 'APOB_IAELSATAQEIIK', 'APOB_SVSLPSLDPASAK', 'APOB_TGISPLALIK', 'APOD_VLNQELR',
               'APOE_AATVGSLAGQPLQER', 'APOE_CLAVYQAGAR', 'APOE_LAVYQAGAR', 'APOE_LGADMEDVR', 'APOE_LGPLVEQGR', 'B2MG_VEHSDLSFSK',
               'B2MG_VNHVTLSQPK', 'B3GN1_EPGEFALLR', 'B3GN1_TALASGGVLDASGDYR', 'B3GN1_YEAAVPDPR', 'BACE1_SIVDSGTTNLR',
               'BASP1_ETPAATEAPSSTPK', 'BTD_LSSGLVTAALYGR', 'BTD_SHLIIAQVAK', 'C1QB_LEQGENVFLQATDK', 'C1QB_VPGLYYFTYHASSR',
               'CA2D1_FVVTDGGITR', 'CA2D1_IKPVFIEDANFGR', 'CA2D1_TASGVNQLVDIYEK', 'CAD13_DIQGSLQDIFK', 'CAD13_INENTGSVSVTR',
               'CAD13_YEVSSPYFK', 'CADM3_EGSVPPLK', 'CADM3_GNPVPQQYLWEK', 'CADM3_SLVTVLGIPQKPIITGYK', 'CAH1_VLDALQAIK', 'CAH1_YSSLAEAASK',
               'CATA_LFAYPDTHR', 'CATD_LVDQNIFSFYLSR', 'CATD_VSTLPAITLK', 'CATD_YSQAVPAVTEGPIPEVLK', 'CATL1_VFQEPLFYEAPR',
               'CCKN_AHLGALLAR', 'CCKN_NLQNLDPSHR', 'CD14_AFPALTSLDLSDNPGLGER', 'CD14_FPAIQNLALR', 'CD14_SWLAELQQWLKPGLK',
               'CD59_AGLQVYNK', 'CERU_IYHSHIDAPK', 'CERU_NNEGTYYSPNYNPQSR', 'CFAB_DAQYAPGYDK', 'CFAB_VSEADSSNADWVTK', 'CFAB_YGLVTYATYPK',
               'CH3L1_ILGQQVPYATK', 'CH3L1_SFTLASSETGVGAPISGPGIPGR', 'CH3L1_VTIDSSYDIAK', 'CLUS_IDSLLENDR', 'CLUS_SGSGLVGR',
               'CLUS_VTTVASHTSDSDVPSGVTEVVVK', 'CMGA_EDSLEAGLPLQVR', 'CMGA_SEALAVDGAGKPGAEEAQDPEGK', 'CMGA_SGEATDGARPQALPEPMQESK',
               'CMGA_SGELEQEEER', 'CMGA_YPGPQAEGDSEGLSQGLVDR', 'CNDP1_ALEQDLPVNIK', 'CNDP1_VFQYIDLHQDEFVQTLK', 'CNDP1_WNYIEGTK',
               'CNTN1_DGEYVVEVR', 'CNTN1_TTKPYPADIVVQFK', 'CNTN2_IIVQAQPEWLK', 'CNTN2_TTGPGGDGIPAEVHIVR', 'CNTN2_VIASNILGTGEPSGPSSK',
               'CNTP2_HELQHPIIAR', 'CNTP2_VDNAPDQQNSHPDLAQEEIR', 'CNTP2_YSSSDWVTQYR', 'CO2_DFHINLFR', 'CO2_HAIILLTDGK',
               'CO2_SSGQWQTPGATR', 'CO3_IHWESASLLR', 'CO3_LSINTHPSQKPLSITVR', 'CO3_TELRPGETLNVNFLLR', 'CO3_TGLQEVEVK',
               'CO3_VPVAVQGEDTVQSLTQGDGVAK', 'CO4A_DHAVDLIQK', 'CO4A_GSFEFPVGDAVSK', 'CO4A_LGQYASPTAK', 'CO4A_NVNFQK',
               'CO4A_VLSLAQEQVGGSPEK', 'CO4A_VTASDPLDTLGSEGALSPGGVASLLR', 'CO5_DINYVNPVIK', 'CO5_TLLPVSKPEIR', 'CO5_VFQFLEK',
               'CO6_ALNHLPLEYNSALYSR', 'CO6_SEYGAALAWEK', 'CO8B_IPGIFELGISSQSDR', 'CO8B_SDLEVAHYK', 'CO8B_YEFILK', 'COCH_GVISNSGGPVR',
               'CRP_ESDTSYVSLK', 'CSTN1_GNLAGLTLR', 'CSTN1_IHGQNVPFDAVVVDK', 'CSTN1_IPDGVVSVSPK', 'CSTN3_ATGEGLIR', 'CSTN3_ESLLLDTTSLQQR',
               'CUTA_TQSSLVPALTDFVR', 'CYTC_ALDFAVGEYNK', 'DAG1_GVHYISVSATR', 'DAG1_LVPVVNNR', 'DAG1_VTIPTDLIASSGDIIK', 'DIAC_ATYIQNYR',
               'ENOG_GNPTVEVDLYTAK', 'ENOG_LGAEVYHTLK', 'ENPP2_SYPEILTLK', 'ENPP2_WWGGQPLWITATK', 'EXTL2_VIVVWNNIGEK', 'FABPH_SIVTLDGGK',
               'FABPH_SLGVGFATR', 'FAM3C_GINVALANGK', 'FAM3C_SALDTAAR', 'FAM3C_SPFEQHIK', 'FAM3C_TGEVLDTK',
               'FBLN1_AITPPHPASQANIIFDITEGNLR', 'FBLN1_IIEVEEEQEDPYLNDR', 'FBLN1_TGYYFDGISR', 'FBLN3_IPSNPSHR', 'FBLN3_LTIIVGPFSF',
               'FBLN3_SGNENGEFYLR', 'FETUA_AHYDLR', 'FETUA_FSVVYAK', 'FETUA_HTLNQIDEVK', 'FMOD_YLPFVPSR', 'GFAP_ALAAELNQLR',
               'GOLM1_DQLVIPDGQEEEQEAAGEGR', 'GOLM1_QQLQALSEPQPR', 'GRIA4_EYPGSETPPK', 'GRIA4_LQNILEQIVSVGK', 'GRIA4_NTDQEYTAFR',
               'HBA_FLASVSTVLTSK', 'HBA_TYFPHFDLSHGSAQVK', 'HBA_VGAHAGEYGAEALER', 'HBB_EFTPPVQAAYQK', 'HBB_SAVTALWGK',
               'HBB_VNVDEVGGEALGR', 'HEMO_NFPSPVDAAFR', 'HEMO_QGHNSVFLIK', 'HEMO_SGAQATWTELPWPHEK', 'I18BP_LWEGSTSR', 'IBP2_HGLYNLK',
               'IBP2_LIQGAPTIR', 'IGSF8_DTQFSYAVFK', 'IGSF8_LQGDAVVLK', 'IGSF8_VVAGEVQVQR', 'ITIH1_EVAFDLEIPK', 'ITIH1_QYYEGSEIVVAGR',
               'ITIH5_SYLEITPSR', 'KAIN_FYYLIASETPGK', 'KAIN_LGFTDLFSK', 'KAIN_VGSALFLSHNLK', 'KAIN_WADLSGITK', 'KLK10_ALQLPYR',
               'KLK11_LPHTLR', 'KLK6_ESSQEQSSVVR', 'KLK6_LSELIQPLPLER', 'KLK6_YTNWIQK', 'KNG1_DIPTNSPELEETLTHTITK', 'KNG1_QVVAGLNFR',
               'KNG1_TVGSDTFYSFK', 'KPYM_LDIDSPPITAR', 'L1CAM_AQLLVVGSPGPVPR', 'L1CAM_LVLSDLHLLTQSQVR', 'L1CAM_WRPVDLAQVK',
               'LAMB2_AQGIAQGAIR', 'LPHN1_LVVSQLNPYTLR', 'LPHN1_SGETVINTANYHDTSPYR', 'LRC4B_HLEILQLSK', 'LRC4B_LTTVPTQAFEYLSK',
               'LTBP2_EQDAPVAGLQPVER', 'MIME_ESAYLYAR', 'MIME_ETVIIPNEK', 'MIME_LEGNPIVLGK', 'MOG_VVHLYR', 'MUC18_EVTVPVFYPTEK',
               'MUC18_GATLALTQVTPQDER', 'NBL1_LALFPDK', 'NCAM1_AGEQDATIHLK', 'NCAM1_GLGEISAASEFK', 'NCAM2_ASGSPEPAISWFR',
               'NCAM2_IIELSQTTAK', 'NCAN_APVLELEK', 'NCAN_LSSAIIAAPR', 'NEGR1_SSIIFAGGDK', 'NEGR1_VVVNFAPTIQEIK', 'NEGR1_WSVDPR',
               'NELL2_AFLFQDTPR', 'NELL2_FTGSSWIK', 'NELL2_SALAYVDGK', 'NEO1_DVVASLVSTR', 'NEUS_ALGITEIFIK', 'NEUS_QEVPLATLEPLVK',
               'NGF_SAPAAAIAAR', 'NICA_ALADVATVLGR', 'NICA_APDVTTLPR', 'NPTX1_FQLTFPLR', 'NPTX1_LENLEQYSR', 'NPTX2_LESLEHQLR',
               'NPTX2_TESTLNALLQR', 'NPTXR_ELDVLQGR', 'NPTXR_LVEAFGGATK', 'NRCAM_SLPSEASEQYLTK', 'NRCAM_VFNTPEGVPSAPSSLK',
               'NRCAM_YIVSGTPTFVPYLIK', 'NRX1A_DLFIDGQSK', 'NRX1A_ITTQITAGAR', 'NRX1A_SDLYIGGVAK', 'NRX2A_AIVADPVTFK',
               'NRX2A_LGERPPALLGSQGLR', 'NRX2A_LSALTLSTVK', 'NRX3A_IYGEVVFK', 'NRX3A_SDLSFQFK', 'OSTP_AIPVAQDLNAPSDWDSR',
               'PCSK1_ALAHLLEAER', 'PCSK1_GEAAGAVQELAR', 'PCSK1_NSDPALGLDDDPDAPAAQLAR', 'PDYN_FLPSISTK', 'PDYN_LSGSFLK',
               'PDYN_SVGEGPYSELAK', 'PEDF_DTDTGALLFIGK', 'PEDF_SSFVAPLEK', 'PEDF_TVQAVLTVPK', 'PGRP2_AGLLRPDYALLGHR', 'PGRP2_TFTLLDPK',
               'PIMT_VQLVVGDGR', 'PLDX1_LYGPSEPHSR', 'PLMN_EAQLPVIENK', 'PLMN_HSIFTPETNPR', 'PLMN_LSSPAVITDK', 'PPN_VHQSPDGTLLIYNLR',
               'PRDX1_DISLSDYK', 'PRDX1_LVQAFQFTDK', 'PRDX2_GLFIIDGK', 'PRDX2_IGKPAPDFK', 'PRDX3_HLSVNDLPVGR', 'PRDX6_LIALSIDSVEDHLAWSK',
               'PRDX6_LSILYPATTGR', 'PTGDS_AQGFTEDTIVFLPQTDK', 'PTGDS_WFSAGLASNSSWLR', 'PTPRN_AEAPALFSR', 'PTPRN_LAAVLAGYGVELR',
               'PTPRN_SELEAQTGLQILQTGVGQR', 'PVRL1_ITQVTWQK', 'SCG1_GEAGAPGEEDIQGPTK', 'SCG1_HLEEPGETQNAFLNER', 'SCG1_NYLNYGEEGAPGK',
               'SCG1_SSQGGSLPSEEK', 'SCG2_ALEYIENLR', 'SCG2_IILEALR', 'SCG2_VLEYLNQEK', 'SCG3_ELSAERPLNEQIAEAEEDK',
               'SCG3_FQDDPDGLHQLDGTPLTAEDIVHK', 'SCG3_LNVEDVDSTK', 'SE6L1_ETGTPIWTSR', 'SE6L1_SPTNTISVYFR', 'SIAE_ELSNTAAYQSVR',
               'SLIK1_SLPVDVFAGVSLSK', 'SODC_GDGPVQGIINFEQK', 'SODC_HVGDLGNVTADK', 'SODC_TLVVHEK', 'SODE_AGLAASLAGPHSIVGR',
               'SODE_AVVVHAGEDDLGR', 'SODE_VTGVVLFR', 'SORC1_TIAVYEEFR', 'SPON1_VTLSAAPPSYFR', 'SPRL1_HIQETEWQSQEGK',
               'SPRL1_HSASDDYFIPSQAFLEAER', 'SPRL1_VLTHSELAPLR', 'TGFB1_LLAPSDSPEWLSFDVTGVVR', 'THRB_ETAASLLQAGYK',
               'THRB_YGFYTHVFR', 'TIMP1_GFQALGDAADIR', 'TIMP1_SEEFLIAGK', 'TNR21_ASNLIGTYR', 'TRFM_ADTDGGLIFR',
               'TTHY_TSESGELHGLTTEEEFVEGIYK', 'TTHY_VEIDTK', 'UBB_ESTLHLVLR', 'UBB_TITLEVEPSDTIENVK', 'UBB_TLSDYNIQK',
               'VASN_NLHDLDVSDNQLER', 'VASN_SLTLGIEPVSPTSLR', 'VASN_YLQGSSVQLR', 'VGF_AYQGVAAPFPK', 'VGF_NSEPQDEGELFQGVDPR',
               'VGF_THLGEALAPLSK', 'VTDB_EFSHLGK', 'VTDB_HLSLLTTLSNR', 'VTDB_VPTADLEDVLPLAEDITNILSK']]
        #df3 = df3[['RID', 'VISCODE2', 'ABETA_RAW', 'ABETA', 'ABETA_y', 'TAU_RAW', 'TAU_1', 'TAU_3', 'TAU_2', 'TAU_4', 'PTAU_RAW', 'PTAU', 'PTAU_y', 'FINAL_ASYN', 'FINAL_PS129', 'FINAL_HGB', 'CTWHITE', 'CTRED', 'PROTEIN', 'GLUCOSE', 'VILIP', 'VILIP_STDEV', 'VILIP_CV', 'YKL', 'YKL_STDEV', 'YKL_CV', 'SNAP', 'SNAP_STDEV', 'SNAP_CV', 'NGRN', 'NGRN_STDEV', 'NGRN_CV', 'WU_STREM2', 'WU_STREM2_CV', 'WU_STREM2CORRECTED', 'MSD_STREM2', 'MSD_STREM2_CV', 'MSD_STREM2CORRECTED', 'MSD_PGRN', 'MSD_PGRN_CV', 'MSD_PGRNCORRECTED']]
        #df3.to_csv('2_tmp/csf.csv', columns=column)
        #print("111111111111111111111111")

    if type == "urine" :
        df3 = df3[['RID', 'VISCODE2', 'ISO8PGF2A', 'ISO812IPF2A']]

    if type == "plasma" :
        #df3 = df3[['RID', 'VISCODE2', 'ANALYSIS_BATCH', 'DDE_RECOVER_13C_PERCENT', 'DDE_PLASMA_CONCENTRATION', 'DETECTION_LIMIT', 'HCAMPLAS', 'HCVOLUME', 'HCRECEIVE', 'HCFROZEN', 'HCRESAMP', 'HCUSABLE', 'PLASMA_NFL', 'PEAK_AREA_ABETA38_N14_B07',	'PEAK_AREA_ABETA38_N14_B08',	'PEAK_AREA_ABETA38_N14_B09',	'PEAK_AREA_ABETA38_N15_B07',	'PEAK_AREA_ABETA38_N15_B08',	'PEAK_AREA_ABETA38_N15_B09',	'PEAK_AREA_ABETA40_N14_B05',	'PEAK_AREA_ABETA40_N14_B07',	'PEAK_AREA_ABETA40_N14_B08',	'PEAK_AREA_ABETA40_N14_B09',	'PEAK_AREA_ABETA40_N14_B10',	'PEAK_AREA_ABETA40_N14_B11',	'PEAK_AREA_ABETA40_N15_B05',	'PEAK_AREA_ABETA40_N15_B07',	'PEAK_AREA_ABETA40_N15_B08',	'PEAK_AREA_ABETA40_N15_B09',	'PEAK_AREA_ABETA40_N15_B10',	'PEAK_AREA_ABETA40_N15_B11',	'PEAK_AREA_ABETA42_N14_B08',	'PEAK_AREA_ABETA42_N14_B09',	'PEAK_AREA_ABETA42_N14_B10',	'PEAK_AREA_ABETA42_N14_B11',	'PEAK_AREA_ABETA42_N14_B12',	'PEAK_AREA_ABETA42_N14_B13',	'PEAK_AREA_ABETA42_N15_B08',	'PEAK_AREA_ABETA42_N15_B09',	'PEAK_AREA_ABETA42_N15_B10',	'PEAK_AREA_ABETA42_N15_B11',	'PEAK_AREA_ABETA42_N15_B12',	'PEAK_AREA_ABETA42_N15_B13',	'PEAK_AREA_ABETAMD_N14_B04',	'PEAK_AREA_ABETAMD_N14_B05',	'PEAK_AREA_ABETAMD_N14_B06',	'PEAK_AREA_ABETAMD_N14_B07',	'PEAK_AREA_ABETAMD_N14_B08',	'PEAK_AREA_ABETAMD_N14_B09',	'PEAK_AREA_ABETAMD_N14_B10',	'PEAK_AREA_ABETAMD_N14_B11',	'PEAK_AREA_ABETAMD_N15_B04',	'PEAK_AREA_ABETAMD_N15_B05',	'PEAK_AREA_ABETAMD_N15_B06',	'PEAK_AREA_ABETAMD_N15_B07',	'PEAK_AREA_ABETAMD_N15_B08',	'PEAK_AREA_ABETAMD_N15_B09',	'PEAK_AREA_ABETAMD_N15_B10',	'PEAK_AREA_ABETAMD_N15_B11',	'PEAK_AREA_ABETA40_N14_TOUSE',	'PEAK_AREA_ABETA40_N15_TOUSE',	'PEAK_AREA_ABETA42_N14_TOUSE',	'PEAK_AREA_ABETA42_N15_TOUSE',	'RATIO_ABETA42_40_BY_ISTD_TOUSE', 'FP40', 'TP40', 'FP42', 'TP42', 'XXL_VLDL_P',	'XXL_VLDL_L',	'XXL_VLDL_PL',	'XXL_VLDL_C',	'XXL_VLDL_CE',	'XXL_VLDL_FC',	'XXL_VLDL_TG',	'XL_VLDL_P',	'XL_VLDL_L',	'XL_VLDL_PL',	'XL_VLDL_C',	'XL_VLDL_CE',	'XL_VLDL_FC',	'XL_VLDL_TG',	'L_VLDL_P',	'L_VLDL_L',	'L_VLDL_PL',	'L_VLDL_C',	'L_VLDL_CE',	'L_VLDL_FC',	'L_VLDL_TG',	'M_VLDL_P',	'M_VLDL_L',	'M_VLDL_PL',	'M_VLDL_C',	'M_VLDL_CE',	'M_VLDL_FC',	'M_VLDL_TG',	'S_VLDL_P',	'S_VLDL_L',	'S_VLDL_PL',	'S_VLDL_C',	'S_VLDL_CE',	'S_VLDL_FC',	'S_VLDL_TG',	'XS_VLDL_P',	'XS_VLDL_L',	'XS_VLDL_PL',	'XS_VLDL_C',	'XS_VLDL_CE',	'XS_VLDL_FC',	'XS_VLDL_TG',	'IDL_P',	'IDL_L',	'IDL_PL',	'IDL_C',	'IDL_CE',	'IDL_FC',	'IDL_TG',	'L_LDL_P',	'L_LDL_L',	'L_LDL_PL',	'L_LDL_C',	'L_LDL_CE',	'L_LDL_FC',	'L_LDL_TG',	'M_LDL_P',	'M_LDL_L',	'M_LDL_PL',	'M_LDL_C',	'M_LDL_CE',	'M_LDL_FC',	'M_LDL_TG',	'S_LDL_P',	'S_LDL_L',	'S_LDL_PL',	'S_LDL_C',	'S_LDL_CE',	'S_LDL_FC',	'S_LDL_TG',	'XL_HDL_P',	'XL_HDL_L',	'XL_HDL_PL',	'XL_HDL_C',	'XL_HDL_CE',	'XL_HDL_FC',	'XL_HDL_TG',	'L_HDL_P',	'L_HDL_L',	'L_HDL_PL',	'L_HDL_C',	'L_HDL_CE',	'L_HDL_FC',	'L_HDL_TG',	'M_HDL_P',	'M_HDL_L',	'M_HDL_PL',	'M_HDL_C',	'M_HDL_CE',	'M_HDL_FC',	'M_HDL_TG',	'S_HDL_P',	'S_HDL_L',	'S_HDL_PL',	'S_HDL_C',	'S_HDL_CE',	'S_HDL_FC',	'S_HDL_TG',	'XXL_VLDL_PL__',	'XXL_VLDL_C__',	'XXL_VLDL_CE__',	'XXL_VLDL_FC__',	'XXL_VLDL_TG__',	'XL_VLDL_PL__',	'XL_VLDL_C__',	'XL_VLDL_CE__',	'XL_VLDL_FC__',	'XL_VLDL_TG__',	'L_VLDL_PL__',	'L_VLDL_C__',	'L_VLDL_CE__',	'L_VLDL_FC__',	'L_VLDL_TG__',	'M_VLDL_PL__',	'M_VLDL_C__',	'M_VLDL_CE__',	'M_VLDL_FC__',	'M_VLDL_TG__',	'S_VLDL_PL__',	'S_VLDL_C__',	'S_VLDL_CE__',	'S_VLDL_FC__',	'S_VLDL_TG__',	'XS_VLDL_PL__',	'XS_VLDL_C__',	'XS_VLDL_CE__',	'XS_VLDL_FC__',	'XS_VLDL_TG__',	'IDL_PL__',	'IDL_C__',	'IDL_CE__',	'IDL_FC__',	'IDL_TG__',	'L_LDL_PL__',	'L_LDL_C__',	'L_LDL_CE__',	'L_LDL_FC__',	'L_LDL_TG__',	'M_LDL_PL__',	'M_LDL_C__',	'M_LDL_CE__',	'M_LDL_FC__',	'M_LDL_TG__',	'S_LDL_PL__',	'S_LDL_C__',	'S_LDL_CE__',	'S_LDL_FC__',	'S_LDL_TG__',	'XL_HDL_PL__',	'XL_HDL_C__',	'XL_HDL_CE__',	'XL_HDL_FC__',	'XL_HDL_TG__',	'L_HDL_PL__',	'L_HDL_C__',	'L_HDL_CE__',	'L_HDL_FC__',	'L_HDL_TG__',	'M_HDL_PL__',	'M_HDL_C__',	'M_HDL_CE__',	'M_HDL_FC__',	'M_HDL_TG__',	'S_HDL_PL__',	'S_HDL_C__',	'S_HDL_CE__',	'S_HDL_FC__',	'S_HDL_TG__',	'VLDL_D',	'LDL_D',	'HDL_D',	'SERUM_C',	'VLDL_C',	'REMNANT_C',	'LDL_C',	'HDL_C',	'HDL2_C',	'HDL3_C',	'ESTC',	'FREEC',	'SERUM_TG',	'VLDL_TG',	'LDL_TG',	'HDL_TG',	'TOTPG',	'TG_PG',	'PC',	'SM',	'TOTCHO',	'APOA1',	'APOB',	'APOB_APOA1',	'TOTFA',	'UNSAT',	'DHA',	'LA',	'FAW3',	'FAW6',	'PUFA',	'MUFA',	'SFA',	'DHA_FA',	'LA_FA',	'FAW3_FA',	'FAW6_FA',	'PUFA_FA',	'MUFA_FA',	'SFA_FA',	'GLC',	'LAC',	'PYR',	'CIT',	'GLOL',	'ALA',	'GLN',	'GLY',	'HIS',	'ILE',	'LEU',	'VAL',	'PHE',	'TYR',	'ACE',	'ACACE',	'BOHBUT',	'CREA',	'ALB',	'GP', 'PLASMATAU', 'AB40', 'AB42',  'Ala',	'Arg',	'Asn',	'Asp',	'Cit',	'Gln',	'Glu',	'Gly',	'His',	'Ile',	'Lys',	'Met',	'Orn',	'Phe',	'Pro',	'Ser',	'Thr',	'Trp',	'Tyr',	'Val',	'Ac.Orn',	'ADMA',	'alpha.AAA',	'c4.OH.Pro',	'canosine',	'Creatinine',	'DOPA',	'Dopamine',	'Histamine',	'Kynurenine',	'Met.So',	'Nitro.Tyr',	'PEA',	'Putrescine',	'Sarcosine',	'Serotonin',	'Spermidine',	'Spermine',	't4.OH.Pro',	'Taurine',	'SDMA', 'CA',	'CDCA',	'DCA',	'GCA',	'GCDCA',	'GDCA',	'GLCA',	'GUDCA',	'TCA',	'TCDCA',	'TDCA',	'TLCA',	'TMCA_A_B',	'TUDCA',	'UDCA',	'CA_CDCA',	'DCA_CA',	'GLCA_CDCA',	'GDCA_CA',	'GDCA_DCA',	'TDCA_CA',	'TLCA_CDCA',	'TDCA_DCA',	'CA_LOGTRANSFORMFLAG',	'CDCA_LOGTRANSFORMFLAG',	'DCA_LOGTRANSFORMFLAG',	'GCA_LOGTRANSFORMFLAG',	'GCDCA_LOGTRANSFORMFLAG',	'GDCA_LOGTRANSFORMFLAG',	'GLCA_LOGTRANSFORMFLAG',	'GUDCA_LOGTRANSFORMFLAG',	'TCA_LOGTRANSFORMFLAG',	'TCDCA_LOGTRANSFORMFLAG',	'TDCA_LOGTRANSFORMFLAG',	'TLCA_LOGTRANSFORMFLAG', 'TMCA_A_B_LOGTRANSFORMFLAG',	'TUDCA_LOGTRANSFORMFLAG',	'UDCA_LOGTRANSFORMFLAG',	'CA_CDCA_LOGTRANSFORMFLAG',	'DCA_CA_LOGTRANSFORMFLAG',	'GLCA_CDCA_LOGTRANSFORMFLAG',	'GDCA_CA_LOGTRANSFORMFLAG',	'GDCA_DCA_LOGTRANSFORMFLAG',	'TDCA_CA_LOGTRANSFORMFLAG',	'TLCA_CDCA_LOGTRANSFORMFLAG',	'TDCA_DCA_LOGTRANSFORMFLAG', 'C0',	'C10',	'C10.1',	'C10.2',	'C12',	'C12.DC',	'C12.1',	'C14',	'C14.1',	'C14.1.OH',	'C14.2',	'C14.2.OH',	'C16',	'C16.OH',	'C16.1',	'C16.1.OH',	'C16.2',	'C16.2.OH',	'C18',	'C18.1',	'C18.1.OH',	'C18.2',	'C2',	'C3',	'C3.DC..C4.OH.',	'C3.OH',	'C3.1',	'C4',	'C4.1',	'C6..C4.1.DC.',	'C5',	'C5.M.DC',	'C5.OH..C3.DC.M.',	'C5.1',	'C5.1.DC',	'C5.DC..C6.OH.',	'C6.1',	'C7.DC',	'C8',	'C9',	'lysoPC.a.C14.0',	'lysoPC.a.C16.0',	'lysoPC.a.C16.1',	'lysoPC.a.C17.0',	'lysoPC.a.C18.0',	'lysoPC.a.C18.1',	'lysoPC.a.C18.2',	'lysoPC.a.C20.3',	'lysoPC.a.C20.4',	'lysoPC.a.C24.0',	'lysoPC.a.C26.0',	'lysoPC.a.C26.1',	'lysoPC.a.C28.0',	'lysoPC.a.C28.1',	'PC.aa.C24.0',	'PC.aa.C26.0',	'PC.aa.C28.1',	'PC.aa.C30.0',	'PC.aa.C32.0',	'PC.aa.C32.1',	'PC.aa.C32.3',	'PC.aa.C34.1',	'PC.aa.C34.2',	'PC.aa.C34.3',	'PC.aa.C34.4',	'PC.aa.C36.0',	'PC.aa.C36.1',	'PC.aa.C36.2',	'PC.aa.C36.3',	'PC.aa.C36.4',	'PC.aa.C36.5',	'PC.aa.C36.6',	'PC.aa.C38.0',	'PC.aa.C38.3',	'PC.aa.C38.4',	'PC.aa.C38.5',	'PC.aa.C38.6',	'PC.aa.C40.1',	'PC.aa.C40.2',	'PC.aa.C40.3',	'PC.aa.C40.4',	'PC.aa.C40.5',	'PC.aa.C40.6',	'PC.aa.C42.0',	'PC.aa.C42.1',	'PC.aa.C42.2',	'PC.aa.C42.4',	'PC.aa.C42.5',	'PC.aa.C42.6',	'PC.ae.C30.0',	'PC.ae.C30.1',	'PC.ae.C30.2',	'PC.ae.C32.1',	'PC.ae.C32.2',	'PC.ae.C34.0',	'PC.ae.C34.1',	'PC.ae.C34.2',	'PC.ae.C34.3',	'PC.ae.C36.0',	'PC.ae.C36.1',	'PC.ae.C36.2',	'PC.ae.C36.3',	'PC.ae.C36.4',	'PC.ae.C36.5',	'PC.ae.C38.0',	'PC.ae.C38.1',	'PC.ae.C38.2',	'PC.ae.C38.3',	'PC.ae.C38.4',	'PC.ae.C38.5',	'PC.ae.C38.6',	'PC.ae.C40.1',	'PC.ae.C40.2',	'PC.ae.C40.3',	'PC.ae.C40.4',	'PC.ae.C40.5',	'PC.ae.C40.6',	'PC.ae.C42.0',	'PC.ae.C42.1',	'PC.ae.C42.2',	'PC.ae.C42.3',	'PC.ae.C42.4',	'PC.ae.C42.5',	'PC.ae.C44.3',	'PC.ae.C44.4',	'PC.ae.C44.5',	'PC.ae.C44.6',	'SM..OH..C14.1',	'SM..OH..C16.1',	'SM..OH..C22.1',	'SM..OH..C22.2',	'SM..OH..C24.1',	'SM.C16.0',	'SM.C16.1',	'SM.C18.0',	'SM.C18.1',	'SM.C20.2',	'SM.C24.0',	'SM.C24.1',	'SM.C26.0',	'SM.C26.1']]
        #df3 = df3.dropna(subset=['AB40'])
        df3 = df3[['RID', 'VISCODE2', 'ANALYSIS_BATCH', 'DDE_RECOVER_13C_PERCENT', 'DDE_PLASMA_CONCENTRATION',
                   'DETECTION_LIMIT', 'HCAMPLAS', 'HCVOLUME', 'HCRECEIVE', 'HCFROZEN', 'HCRESAMP', 'HCUSABLE',
                   'PLASMA_NFL','FP40', 'TP40', 'FP42', 'TP42', 'XXL_VLDL_P', 'XXL_VLDL_L', 'XXL_VLDL_PL', 'XXL_VLDL_C',
                   'XXL_VLDL_CE', 'XXL_VLDL_FC', 'XXL_VLDL_TG', 'XL_VLDL_P', 'XL_VLDL_L', 'XL_VLDL_PL', 'XL_VLDL_C',
                   'XL_VLDL_CE', 'XL_VLDL_FC', 'XL_VLDL_TG', 'L_VLDL_P', 'L_VLDL_L', 'L_VLDL_PL', 'L_VLDL_C',
                   'L_VLDL_CE', 'L_VLDL_FC', 'L_VLDL_TG', 'M_VLDL_P', 'M_VLDL_L', 'M_VLDL_PL', 'M_VLDL_C', 'M_VLDL_CE',
                   'M_VLDL_FC', 'M_VLDL_TG', 'S_VLDL_P', 'S_VLDL_L', 'S_VLDL_PL', 'S_VLDL_C', 'S_VLDL_CE', 'S_VLDL_FC',
                   'S_VLDL_TG', 'XS_VLDL_P', 'XS_VLDL_L', 'XS_VLDL_PL', 'XS_VLDL_C', 'XS_VLDL_CE', 'XS_VLDL_FC',
                   'XS_VLDL_TG', 'IDL_P', 'IDL_L', 'IDL_PL', 'IDL_C', 'IDL_CE', 'IDL_FC', 'IDL_TG', 'L_LDL_P',
                   'L_LDL_L', 'L_LDL_PL', 'L_LDL_C', 'L_LDL_CE', 'L_LDL_FC', 'L_LDL_TG', 'M_LDL_P', 'M_LDL_L',
                   'M_LDL_PL', 'M_LDL_C', 'M_LDL_CE', 'M_LDL_FC', 'M_LDL_TG', 'S_LDL_P', 'S_LDL_L', 'S_LDL_PL',
                   'S_LDL_C', 'S_LDL_CE', 'S_LDL_FC', 'S_LDL_TG', 'XL_HDL_P', 'XL_HDL_L', 'XL_HDL_PL', 'XL_HDL_C',
                   'XL_HDL_CE', 'XL_HDL_FC', 'XL_HDL_TG', 'L_HDL_P', 'L_HDL_L', 'L_HDL_PL', 'L_HDL_C', 'L_HDL_CE',
                   'L_HDL_FC', 'L_HDL_TG', 'M_HDL_P', 'M_HDL_L', 'M_HDL_PL', 'M_HDL_C', 'M_HDL_CE', 'M_HDL_FC',
                   'M_HDL_TG', 'S_HDL_P', 'S_HDL_L', 'S_HDL_PL', 'S_HDL_C', 'S_HDL_CE', 'S_HDL_FC', 'S_HDL_TG',
                   'XXL_VLDL_PL__', 'XXL_VLDL_C__', 'XXL_VLDL_CE__', 'XXL_VLDL_FC__', 'XXL_VLDL_TG__', 'XL_VLDL_PL__',
                   'XL_VLDL_C__', 'XL_VLDL_CE__', 'XL_VLDL_FC__', 'XL_VLDL_TG__', 'L_VLDL_PL__', 'L_VLDL_C__',
                   'L_VLDL_CE__', 'L_VLDL_FC__', 'L_VLDL_TG__', 'M_VLDL_PL__', 'M_VLDL_C__', 'M_VLDL_CE__',
                   'M_VLDL_FC__', 'M_VLDL_TG__', 'S_VLDL_PL__', 'S_VLDL_C__', 'S_VLDL_CE__', 'S_VLDL_FC__',
                   'S_VLDL_TG__', 'XS_VLDL_PL__', 'XS_VLDL_C__', 'XS_VLDL_CE__', 'XS_VLDL_FC__', 'XS_VLDL_TG__',
                   'IDL_PL__', 'IDL_C__', 'IDL_CE__', 'IDL_FC__', 'IDL_TG__', 'L_LDL_PL__', 'L_LDL_C__', 'L_LDL_CE__',
                   'L_LDL_FC__', 'L_LDL_TG__', 'M_LDL_PL__', 'M_LDL_C__', 'M_LDL_CE__', 'M_LDL_FC__', 'M_LDL_TG__',
                   'S_LDL_PL__', 'S_LDL_C__', 'S_LDL_CE__', 'S_LDL_FC__', 'S_LDL_TG__', 'XL_HDL_PL__', 'XL_HDL_C__',
                   'XL_HDL_CE__', 'XL_HDL_FC__', 'XL_HDL_TG__', 'L_HDL_PL__', 'L_HDL_C__', 'L_HDL_CE__', 'L_HDL_FC__',
                   'L_HDL_TG__', 'M_HDL_PL__', 'M_HDL_C__', 'M_HDL_CE__', 'M_HDL_FC__', 'M_HDL_TG__', 'S_HDL_PL__',
                   'S_HDL_C__', 'S_HDL_CE__', 'S_HDL_FC__', 'S_HDL_TG__', 'VLDL_D', 'LDL_D', 'HDL_D', 'SERUM_C',
                   'VLDL_C', 'REMNANT_C', 'LDL_C', 'HDL_C', 'HDL2_C', 'HDL3_C', 'ESTC', 'FREEC', 'SERUM_TG', 'VLDL_TG',
                   'LDL_TG', 'HDL_TG', 'TOTPG', 'TG_PG', 'PC', 'SM', 'TOTCHO', 'APOA1', 'APOB', 'APOB_APOA1', 'TOTFA',
                   'UNSAT', 'DHA', 'LA', 'FAW3', 'FAW6', 'PUFA', 'MUFA', 'SFA', 'DHA_FA', 'LA_FA', 'FAW3_FA', 'FAW6_FA',
                   'PUFA_FA', 'MUFA_FA', 'SFA_FA', 'GLC', 'LAC', 'PYR', 'CIT', 'GLOL', 'ALA', 'GLN', 'GLY', 'HIS',
                   'ILE', 'LEU', 'VAL', 'PHE', 'TYR', 'ACE', 'ACACE', 'BOHBUT', 'CREA', 'ALB', 'GP', 'PLASMATAU',
                   'AB40', 'AB42', 'Ala', 'Arg', 'Asn', 'Asp', 'Cit', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 'Lys', 'Met',
                   'Orn', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val', 'Ac.Orn', 'ADMA', 'alpha.AAA', 'c4.OH.Pro',
                   'canosine', 'Creatinine', 'DOPA', 'Dopamine', 'Histamine', 'Kynurenine', 'Met.So', 'Nitro.Tyr',
                   'PEA', 'Putrescine', 'Sarcosine', 'Serotonin', 'Spermidine', 'Spermine', 't4.OH.Pro', 'Taurine',
                   'SDMA', 'CA', 'CDCA', 'DCA', 'GCA', 'GCDCA', 'GDCA', 'GLCA', 'GUDCA', 'TCA', 'TCDCA', 'TDCA', 'TLCA',
                   'TMCA_A_B', 'TUDCA', 'UDCA', 'CA_CDCA', 'DCA_CA', 'GLCA_CDCA', 'GDCA_CA', 'GDCA_DCA', 'TDCA_CA',
                   'TLCA_CDCA', 'TDCA_DCA', 'CA_LOGTRANSFORMFLAG', 'CDCA_LOGTRANSFORMFLAG', 'DCA_LOGTRANSFORMFLAG',
                   'GCA_LOGTRANSFORMFLAG', 'GCDCA_LOGTRANSFORMFLAG', 'GDCA_LOGTRANSFORMFLAG', 'GLCA_LOGTRANSFORMFLAG',
                   'GUDCA_LOGTRANSFORMFLAG', 'TCA_LOGTRANSFORMFLAG', 'TCDCA_LOGTRANSFORMFLAG', 'TDCA_LOGTRANSFORMFLAG',
                   'TLCA_LOGTRANSFORMFLAG', 'TMCA_A_B_LOGTRANSFORMFLAG', 'TUDCA_LOGTRANSFORMFLAG',
                   'UDCA_LOGTRANSFORMFLAG', 'CA_CDCA_LOGTRANSFORMFLAG', 'DCA_CA_LOGTRANSFORMFLAG',
                   'GLCA_CDCA_LOGTRANSFORMFLAG', 'GDCA_CA_LOGTRANSFORMFLAG', 'GDCA_DCA_LOGTRANSFORMFLAG',
                   'TDCA_CA_LOGTRANSFORMFLAG', 'TLCA_CDCA_LOGTRANSFORMFLAG', 'TDCA_DCA_LOGTRANSFORMFLAG', 'C0', 'C10',
                   'C10.1', 'C10.2', 'C12', 'C12.DC', 'C12.1', 'C14', 'C14.1', 'C14.1.OH', 'C14.2', 'C14.2.OH', 'C16',
                   'C16.OH', 'C16.1', 'C16.1.OH', 'C16.2', 'C16.2.OH', 'C18', 'C18.1', 'C18.1.OH', 'C18.2', 'C2', 'C3',
                   'C3.DC..C4.OH.', 'C3.OH', 'C3.1', 'C4', 'C4.1', 'C6..C4.1.DC.', 'C5', 'C5.M.DC', 'C5.OH..C3.DC.M.',
                   'C5.1', 'C5.1.DC', 'C5.DC..C6.OH.', 'C6.1', 'C7.DC', 'C8', 'C9', 'lysoPC.a.C14.0', 'lysoPC.a.C16.0',
                   'lysoPC.a.C16.1', 'lysoPC.a.C17.0', 'lysoPC.a.C18.0', 'lysoPC.a.C18.1', 'lysoPC.a.C18.2',
                   'lysoPC.a.C20.3', 'lysoPC.a.C20.4', 'lysoPC.a.C24.0', 'lysoPC.a.C26.0', 'lysoPC.a.C26.1',
                   'lysoPC.a.C28.0', 'lysoPC.a.C28.1', 'PC.aa.C24.0', 'PC.aa.C26.0', 'PC.aa.C28.1', 'PC.aa.C30.0',
                   'PC.aa.C32.0', 'PC.aa.C32.1', 'PC.aa.C32.3', 'PC.aa.C34.1', 'PC.aa.C34.2', 'PC.aa.C34.3',
                   'PC.aa.C34.4', 'PC.aa.C36.0', 'PC.aa.C36.1', 'PC.aa.C36.2', 'PC.aa.C36.3', 'PC.aa.C36.4',
                   'PC.aa.C36.5', 'PC.aa.C36.6', 'PC.aa.C38.0', 'PC.aa.C38.3', 'PC.aa.C38.4', 'PC.aa.C38.5',
                   'PC.aa.C38.6', 'PC.aa.C40.1', 'PC.aa.C40.2', 'PC.aa.C40.3', 'PC.aa.C40.4', 'PC.aa.C40.5',
                   'PC.aa.C40.6', 'PC.aa.C42.0', 'PC.aa.C42.1', 'PC.aa.C42.2', 'PC.aa.C42.4', 'PC.aa.C42.5',
                   'PC.aa.C42.6', 'PC.ae.C30.0', 'PC.ae.C30.1', 'PC.ae.C30.2', 'PC.ae.C32.1', 'PC.ae.C32.2',
                   'PC.ae.C34.0', 'PC.ae.C34.1', 'PC.ae.C34.2', 'PC.ae.C34.3', 'PC.ae.C36.0', 'PC.ae.C36.1',
                   'PC.ae.C36.2', 'PC.ae.C36.3', 'PC.ae.C36.4', 'PC.ae.C36.5', 'PC.ae.C38.0', 'PC.ae.C38.1',
                   'PC.ae.C38.2', 'PC.ae.C38.3', 'PC.ae.C38.4', 'PC.ae.C38.5', 'PC.ae.C38.6', 'PC.ae.C40.1',
                   'PC.ae.C40.2', 'PC.ae.C40.3', 'PC.ae.C40.4', 'PC.ae.C40.5', 'PC.ae.C40.6', 'PC.ae.C42.0',
                   'PC.ae.C42.1', 'PC.ae.C42.2', 'PC.ae.C42.3', 'PC.ae.C42.4', 'PC.ae.C42.5', 'PC.ae.C44.3',
                   'PC.ae.C44.4', 'PC.ae.C44.5', 'PC.ae.C44.6', 'SM..OH..C14.1', 'SM..OH..C16.1', 'SM..OH..C22.1',
                   'SM..OH..C22.2', 'SM..OH..C24.1', 'SM.C16.0', 'SM.C16.1', 'SM.C18.0', 'SM.C18.1', 'SM.C20.2',
                   'SM.C24.0', 'SM.C24.1', 'SM.C26.0', 'SM.C26.1']]

    if type == "serum" :
        df3 = df3[['RID', 'VISCODE2', 'CA', 'GCA', 'GDCA', 'Serotonin', 'Sarcosine', 'Kynurenine', 'Taurine', 'C0', 'C2', 'C3', 'C10', 'C18']]
        #df3 = df3.dropna(subset=['AB40'])
        print("2222")


    df3 = df3.drop_duplicates()
    df3 = drop_col(df3)
    #print(df1)

    df3 = drop_row(df3)
    #print(df3)
    if not os.path.exists(SaveFile_Path):
        os.makedirs(SaveFile_Path)
    df3.to_csv(SaveFile_Path + '/' + SaveFile_Name, index = False)


def main() :
    csf_clear_data()
    fill_up_viscode2(Csf_Path)
    csv_merge('csf', Csf_Path, Save_Csf_Path, Save_Csf_Name)
    urine_clear_data()
    fill_up_viscode2(Urine_Path)
    csv_merge('urine', Urine_Path, Save_Serum_Path, Save_Urine_Name)
    plasma_clear_data()
    fill_up_viscode2(Plasma_Path)
    csv_merge('plasma', Plasma_Path, Save_Plasma_Path, Save_Plasma_Name)
    #serum_clear_data()
    #fill_up_viscode2(Serum_Path)
    #csv_merge('serum', Serum_Path, Save_Serum_Path, Save_Serum_Name)
    #print_first_line()

if __name__ == '__main__':
    main()


