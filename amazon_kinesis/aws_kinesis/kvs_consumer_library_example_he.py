# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0.

'''
Example to demonstrate usage the AWS Kinesis Video Streams (KVS) Consumer Library for Python.
 '''
 
__version__ = "0.0.1"
__status__ = "Development"
__copyright__ = "Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved."
__author__ = "Dean Colcott <https://www.linkedin.com/in/deancolcott/>"

import os
import sys
import time
import boto3
import logging
from amazon_kinesis_video_consumer_library.kinesis_video_streams_parser import KvsConsumerLibrary
from amazon_kinesis_video_consumer_library.kinesis_video_fragment_processor import KvsFragementProcessor

# Config the logger.
log = logging.getLogger(__name__)
logging.basicConfig(format="[%(name)s.%(funcName)s():%(lineno)d] - [%(levelname)s] - %(message)s", 
                    stream=sys.stdout, 
                    level=logging.INFO)

# Update the desired region and KVS stream name.
REGION='ap-northeast-2'
KVS_STREAM01_NAME = 'Ju-connect-catholic-test-contact-059dd53c-13de-4a5f-99b8-2beb715afe16'   # Stream must be in specified region


class KvsPythonConsumerExample:
    def __init__(self):
        self.kvs_fragment_processor = KvsFragementProcessor()
        self.last_good_fragment_tags = None
        log.info('Initializing Amazon Kinesis Video client....')
        self.session = boto3.Session(region_name=REGION)
        self.kvs_client = self.session.client("kinesisvideo")

    ####################################################
    # Main process loop
    def service_loop(self):
        
        ####################################################
        # Start an instance of the KvsConsumerLibrary reading in a Kinesis Video Stream

        # Get the KVS Endpoint for the GetMedia Call for this stream
        log.info(f'Getting KVS GetMedia Endpoint for stream: {KVS_STREAM01_NAME} ........') 
        get_media_endpoint = self._get_data_endpoint(KVS_STREAM01_NAME, 'GET_MEDIA')  #엔드포인트 가져오기
        
        # Get the KVS Media client for the GetMedia API call
        log.info(f'Initializing KVS Media client for stream: {KVS_STREAM01_NAME}........') 
        kvs_media_client = self.session.client('kinesis-video-media', endpoint_url=get_media_endpoint)   # KVS 미디어 클라이언트 초기화

        # Make a KVS GetMedia API call with the desired KVS stream and StartSelector type and time bounding.
        log.info(f'Requesting KVS GetMedia Response for stream: {KVS_STREAM01_NAME}........') 
        get_media_response = kvs_media_client.get_media(  # GetMedia 응답 받기. 스트림에서 미디어 데이터를 가져오기
            StreamName=KVS_STREAM01_NAME,
            StartSelector={
                'StartSelectorType': 'NOW'
            }
        )

        # Initialize an instance of the KvsConsumerLibrary, provide the GetMedia response and the required call-backs
        # 프래그먼트를 처리하기 위한 콜백 함수를 제공하여 KvsConsumerLibrary 인스턴스를 생성
        log.info(f'Starting KvsConsumerLibrary for stream: {KVS_STREAM01_NAME}........') 
        my_stream01_consumer = KvsConsumerLibrary(KVS_STREAM01_NAME, 
                                              get_media_response, 
                                              self.on_fragment_arrived, 
                                              self.on_stream_read_complete, 
                                              self.on_stream_read_exception
                                            )

        # Start the instance of KvsConsumerLibrary, any matching fragments will begin arriving in the on_fragment_arrived callback
        my_stream01_consumer.start()
        while True:

            #Add Main process / application logic here while KvsConsumerLibrary instance runs as a thread
            log.info("Nothn to see, just doin main application stuff in a loop here!")
            time.sleep(5)


    ####################################################
    # KVS Consumer Library call-backs
   
    def on_fragment_arrived(self, stream_name, fragment_bytes, fragment_dom, fragment_receive_duration):  # 프래그먼트가 도착했을 때 호출되는 콜백 함수
        try:
            log.info(f'\n\n##########################\nFragment Received on Stream: {stream_name}\n##########################')
            log.info('')
            log.info(f'####### Fragment Receive and Processing Duration: {fragment_receive_duration} Secs')

            # Get the fragment tags and save in local parameter.
            self.last_good_fragment_tags = self.kvs_fragment_processor.get_fragment_tags(fragment_dom)

            ##### Log Time Deltas:  local time Vs fragment SERVER and PRODUCER Timestamp:
            time_now = time.time()
            kvs_ms_behind_live = float(self.last_good_fragment_tags['AWS_KINESISVIDEO_MILLIS_BEHIND_NOW'])
            producer_timestamp = float(self.last_good_fragment_tags['AWS_KINESISVIDEO_PRODUCER_TIMESTAMP'])
            server_timestamp = float(self.last_good_fragment_tags['AWS_KINESISVIDEO_SERVER_TIMESTAMP'])
            
            log.info('')
            log.info('####### Timestamps and Delta: ')
            log.info(f'KVS Reported Time Behind Live {kvs_ms_behind_live} mS')
            log.info(f'Local Time Diff to Fragment Producer Timestamp: {round(((time_now - producer_timestamp)*1000), 3)} mS')
            log.info(f'Local Time Diff to Fragment Server Timestamp: {round(((time_now - server_timestamp)*1000), 3)} mS')

            ###########################################
            # 1) Extract and print the MKV Tags in the fragment
            ###########################################
            log.info('')
            log.info('####### Fragment MKV Tags:')
            for key, value in self.last_good_fragment_tags.items():
                log.info(f'{key} : {value}')

            ###########################################
            # 2) Pretty Print the entire fragment DOM structure
            # ###########################################
            log.info('')
            log.info('####### Pretty Print Fragment DOM: #######')
            pretty_frag_dom = self.kvs_fragment_processor.get_fragement_dom_pretty_string(fragment_dom)
            log.info(pretty_frag_dom)

            ###########################################
            # 3) Write the Fragment to disk as standalone MKV file
            ###########################################
            save_dir = 'ENTER_DIRECTORY_PATH_TO_SAVE_FRAGEMENTS'
            frag_file_name = self.last_good_fragment_tags['AWS_KINESISVIDEO_FRAGMENT_NUMBER'] + '.mkv' # Update as needed
            frag_file_path = os.path.join(save_dir, frag_file_name)

            ###########################################
            # 4) Extract Frames from Fragment as ndarrays:
            ###########################################
            one_in_frames_ratio = 5
            log.info('')
            log.info(f'#######  Reading 1 in {one_in_frames_ratio} Frames from fragment as ndarray:')
            ndarray_frames = self.kvs_fragment_processor.get_frames_as_ndarray(fragment_bytes, one_in_frames_ratio)
            for i in range(len(ndarray_frames)):
                ndarray_frame = ndarray_frames[i]
                log.info(f'Frame-{i} Shape: {ndarray_frame.shape}')
            
            ###########################################
            # 5) Save Frames from Fragment to local disk as JPGs
            ###########################################
            # one_in_frames_ratio = 5
            # save_dir = 'ENTER_DIRECTORY_PATH_TO_SAVE_JPEG_FRAMES'
            # jpg_file_base_name = self.last_good_fragment_tags['AWS_KINESISVIDEO_FRAGMENT_NUMBER']
            # jpg_file_base_path = os.path.join(save_dir, jpg_file_base_name)
            
            ###########################################
            # 6) Save Amazon Connect Frames from Fragment to local disk as WAVs
            ###########################################
            save_dir = '/home/user09/beaver/data/shared_files/amazon_kinesis/aws_kinesis/'
            wav_file_base_name = self.last_good_fragment_tags['AWS_KINESISVIDEO_FRAGMENT_NUMBER']
            wav_file_base_path = os.path.join(save_dir, wav_file_base_name)
            log.info('')
            log.info(f'####### Saving audio track "AUDIO_FROM_CUSTOMER" from Amazon Connect fragment as WAV to base path: {wav_file_base_path}')
            self.kvs_fragment_processor.save_connect_fragment_audio_track_from_customer_as_wav(fragment_dom, wav_file_base_path)
            log.info(f'####### 저장 audio track "AUDIO_TO_CUSTOMER" from Amazon Connect fragment as WAV to base path: {wav_file_base_path}')
            self.kvs_fragment_processor.save_connect_fragment_audio_track_to_customer_as_wav(fragment_dom, wav_file_base_path)


        except Exception as err:
            log.error(f'on_fragment_arrived Error: {err}')
    
    def on_stream_read_complete(self, stream_name):
        # Do something here to tell the application that reading from the stream ended gracefully.
        print(f'Read Media on stream: {stream_name} Completed successfully - Last Fragment Tags: {self.last_good_fragment_tags}')

    def on_stream_read_exception(self, stream_name, error):
        # Here we just log the error 
        print(f'####### ERROR: Exception on read stream: {stream_name}\n####### Fragment Tags:\n{self.last_good_fragment_tags}\nError Message:{error}')

    ####################################################
    # KVS Helpers
    def _get_data_endpoint(self, stream_name, api_name):
        '''
        Convenience method to get the KVS client endpoint for specific API calls. 
        '''
        response = self.kvs_client.get_data_endpoint(
            StreamName=stream_name,
            APIName=api_name
        )
        return response['DataEndpoint']

if __name__ == "__main__":
    '''
    Main method for example KvsConsumerLibrary
    '''
    
    kvsConsumerExample = KvsPythonConsumerExample()
    kvsConsumerExample.service_loop()

