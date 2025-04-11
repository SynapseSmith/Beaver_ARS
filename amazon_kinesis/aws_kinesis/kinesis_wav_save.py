import os
import boto3
import logging
from amazon_kinesis.aws_kinesis.amazon_kinesis_video_consumer_library.kinesis_video_fragment_processor import KvsFragementProcessor
from amazon_kinesis.aws_kinesis.amazon_kinesis_video_consumer_library.kinesis_video_streams_parser import KvsConsumerLibrary
from flask import jsonify

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# KVS 관련 설정
REGION = '[ENTER_REGION]'
KVS_STREAM_NAME = '[ENTER_KVS_STREAM_NAME]'

# KvsFragmentProcessor 인스턴스 생성
kvs_processor = KvsFragementProcessor()

# Lambda를 호출하여 프래그먼트를 처리하고, WAV 파일로 저장
def lambda_handler(event, context):
    try:
        # Kinesis Video Streams 클라이언트 초기화
        kvs_client = boto3.client('kinesisvideo', region_name=REGION)
        
        # GetMedia API 호출하여 스트림에서 데이터를 가져오기 위한 Endpoint 획득
        logger.info(f'Getting KVS GetMedia Endpoint for stream: {KVS_STREAM_NAME}...')
        response = kvs_client.get_data_endpoint(
            StreamName=KVS_STREAM_NAME,
            APIName='GET_MEDIA'
        )
        endpoint_url = response['DataEndpoint']
        
        # Kinesis Video Media 클라이언트 초기화
        kvs_media_client = boto3.client('kinesis-video-media', endpoint_url=endpoint_url, region_name=REGION)

        # GetMedia API로 스트림 데이터 가져오기
        logger.info(f'Requesting KVS GetMedia response for stream: {KVS_STREAM_NAME}...')
        get_media_response = kvs_media_client.get_media(
            StreamName=KVS_STREAM_NAME,
            StartSelector={'StartSelectorType': 'NOW'}
        )

        # 프래그먼트 처리 (프래그먼트를 처리하면서 프래그먼트를 WAV로 저장)
        process_fragments(get_media_response)
        
        return jsonify({"status": "success"}), 200

    except Exception as e:
        logger.error(f'Error in processing Kinesis stream: {str(e)}')
        return jsonify({"error": str(e)}), 500


def process_fragments(get_media_response):
    """
    프래그먼트를 받아서 처리하고, WAV 파일로 저장하는 함수.
    """
    # Kinesis Video Stream에서 가져온 프래그먼트를 처리하기 위한 소비자 라이브러리
    stream_consumer = KvsConsumerLibrary(
        KVS_STREAM_NAME,
        get_media_response,
        on_fragment_arrived,  # 프래그먼트가 도착하면 호출할 콜백 함수
        on_stream_read_complete,
        on_stream_read_exception
    )
    
    # 프래그먼트 처리 시작
    stream_consumer.start()


def on_fragment_arrived(stream_name, fragment_bytes, fragment_dom, fragment_receive_duration):
    """
    프래그먼트가 도착하면 호출되는 콜백 함수. 프래그먼트를 WAV 파일로 저장.
    """
    try:
        logger.info(f'Fragment arrived for stream: {stream_name}')
        
        # 프래그먼트를 WAV 파일로 저장
        save_dir = "/tmp"  # Lambda 함수에서는 /tmp 디렉토리 사용
        wav_file_name = f"{fragment_dom['AWS_KINESISVIDEO_FRAGMENT_NUMBER']}.wav"
        wav_file_path = os.path.join(save_dir, wav_file_name)
        
        # 프래그먼트를 저장하기 위한 KvsFragmentProcessor 사용
        kvs_processor.save_connect_fragment_audio_track_to_customer_as_wav(fragment_dom, wav_file_path)
        
        logger.info(f'WAV file saved: {wav_file_path}')
    
    except Exception as e:
        logger.error(f'Error in processing fragment: {str(e)}')


def on_stream_read_complete(stream_name):
    """
    스트림 읽기가 완료되었을 때 호출되는 콜백 함수.
    """
    logger.info(f'Reading stream {stream_name} completed.')


def on_stream_read_exception(stream_name, error):
    """
    스트림 읽기 도중 예외가 발생했을 때 호출되는 콜백 함수.
    """
    logger.error(f'Error while reading stream {stream_name}: {error}')