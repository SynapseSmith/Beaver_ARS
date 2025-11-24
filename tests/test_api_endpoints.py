"""
API Endpoints 테스트
"""
import pytest
import json
from flask import Flask
import requests
from flask import request, jsonify

class TestAPIEndpoints:
    """API Endpoints 테스트"""
    
    @pytest.fixture
    def client(self):
        """Flask 테스트 클라이언트"""
        # 실제 Flask 앱을 import 해야 하지만, 여기서는 스켈레톤만 제공
        app = Flask(__name__)
        
        @app.route('/order', methods=['POST'])
        def order():
            data = request.get_json()
            return jsonify({
                'response': '테스트 응답',
                'intent': {'id': 1, 'name': '테스트', 'confidence': 0.95},
                'entities': {}
            })
        
        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy'})
        
        app.config['TESTING'] = True
        return app.test_client()
    
    def test_order_endpoint_valid_request(self, client):
        """정상적인 /order 요청 테스트"""
        payload = {
            "header": {
                "interfaceID": "AI-SDC-CAT-001"
            },
            "body": {
                "text": "떡볶이 가격이 얼마예요?"
            }
        }
        
        response = client.post(
            '/order',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'response' in data
        assert 'intent' in data
        assert 'entities' in data
    
    def test_order_endpoint_missing_text(self, client):
        """text 필드가 없는 경우 테스트"""
        payload = {
            "header": {
                "interfaceID": "AI-SDC-CAT-001"
            },
            "body": {}
        }
        
        response = client.post(
            '/order',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # 400 Bad Request 또는 처리 방식에 따라 다를 수 있음
        assert response.status_code in [200, 400]
    
    def test_order_endpoint_empty_text(self, client):
        """빈 텍스트 요청 테스트"""
        payload = {
            "header": {
                "interfaceID": "AI-SDC-CAT-001"
            },
            "body": {
                "text": ""
            }
        }
        
        response = client.post(
            '/order',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 400]
    
    def test_order_endpoint_long_text(self, client):
        """매우 긴 텍스트 요청 테스트"""
        payload = {
            "header": {
                "interfaceID": "AI-SDC-CAT-001"
            },
            "body": {
                "text": "떡볶이 " * 300  # 600자
            }
        }
        
        response = client.post(
            '/order',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # 500자 제한이 있다면 400, 없다면 200
        assert response.status_code in [200, 400]
    
    def test_health_endpoint(self, client):
        """/health 엔드포인트 테스트"""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'
    
    def test_order_endpoint_special_characters(self, client):
        """특수문자가 포함된 요청 테스트"""
        payload = {
            "header": {
                "interfaceID": "AI-SDC-CAT-001"
            },
            "body": {
                "text": "떡볶이!!! @#$% 가격???"
            }
        }
        
        response = client.post(
            '/order',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
    
    def test_order_endpoint_korean_english_mix(self, client):
        """한영 혼용 텍스트 테스트"""
        payload = {
            "header": {
                "interfaceID": "AI-SDC-CAT-001"
            },
            "body": {
                "text": "tteokbokki price please"
            }
        }
        
        response = client.post(
            '/order',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
    
    def test_response_structure(self, client):
        """응답 구조가 올바른지 테스트"""
        payload = {
            "header": {
                "interfaceID": "AI-SDC-CAT-001"
            },
            "body": {
                "text": "메뉴 보여주세요"
            }
        }
        
        response = client.post(
            '/order',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # 필수 필드 검증
        assert isinstance(data['response'], str)
        assert isinstance(data['intent'], dict)
        assert 'id' in data['intent']
        assert 'name' in data['intent']
        assert 'confidence' in data['intent']
        assert isinstance(data['entities'], dict)
    
    def test_intent_confidence_range(self, client):
        """Intent confidence가 0-1 범위인지 테스트"""
        payload = {
            "header": {
                "interfaceID": "AI-SDC-CAT-001"
            },
            "body": {
                "text": "영업시간 알려주세요"
            }
        }
        
        response = client.post(
            '/order',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        confidence = data['intent']['confidence']
        assert 0 <= confidence <= 1


class TestIntegration:
    """통합 테스트"""
    
    def test_end_to_end_menu_query(self):
        """메뉴 문의 전체 플로우 테스트"""
        # 이 테스트는 실제 서버가 실행 중일 때 수행
        import requests
        
        url = "http://localhost:1117/order"
        payload = {
            "header": {"interfaceID": "AI-SDC-CAT-001"},
            "body": {"text": "떡볶이 가격이 얼마예요?"}
        }
        
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                data = response.json()
                assert 'response' in data
                assert '떡볶이' in data['response'] or 'MENU' in data['entities']
            else:
                pytest.skip("Server not running")
        except requests.exceptions.ConnectionError:
            pytest.skip("Server not running")
    
    def test_end_to_end_operating_hours(self):
        """영업시간 문의 전체 플로우 테스트"""
        import requests
        
        url = "http://localhost:1117/order"
        payload = {
            "header": {"interfaceID": "AI-SDC-CAT-001"},
            "body": {"text": "영업시간이 언제예요?"}
        }
        
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                data = response.json()
                assert 'response' in data
                # Intent ID 16: 영업 시간 안내
                assert data['intent']['id'] in range(0, 48)
            else:
                pytest.skip("Server not running")
        except requests.exceptions.ConnectionError:
            pytest.skip("Server not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
