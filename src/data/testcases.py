import warnings
warnings.simplefilter(action='ignore', category=Warning)
from controller_getdata_insertDB import ControllerClass
import logging
import pandas


def main():
    logger = logging.getLogger(__name__)
    logger.info('Start Logging')
    obj = ControllerClass('AndroidPerf.db')
    logger.info('Perform test case to insert in db')
    #obj.getdata_insertdb('tv.airtel.smartstick', 'Airtel_PinScreen')
    #obj.getdata_insertdb('tv.airtel.smartstick', 'Airtel_HomeScreen')
    #obj.getdata_insertdb('tv.airtel.smartstick', 'Airtel_DetailScreen')
    #obj.getdata_insertdb('tv.airtel.smartstick', 'Airtel_PlayerScreen')
    #obj.getdata_insertdb('tv.airtel.smartstick', 'Airtel_Scrolling')
    #obj.getdata_insertdb('tv.airtel.smartstick', 'Airtel_SearchScreen')

    #obj.getdata_insertdb('com.netflix.ninja', 'Netflix_PinScreen')
    #obj.getdata_insertdb('com.netflix.ninja', 'Netflix_HomeScreen')
    #obj.getdata_insertdb('com.netflix.ninja', 'Netflix_DetailScreen')
    #obj.getdata_insertdb('com.netflix.ninja', 'Netflix_PlayerScreen')
    #obj.getdata_insertdb('com.netflix.ninja', 'Netflix_Scrolling')
    #obj.getdata_insertdb('com.netflix.ninja', 'Netflix_SearchScreen')

    logger.info('Data inserted in DB')
    print obj.fetch_Dataframe()

    obj.closeDB()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
