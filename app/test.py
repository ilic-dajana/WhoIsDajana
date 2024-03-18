
import unittest
from unittest.mock import MagicMock

from main import APP

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = APP.test_client()
        self.app.testing = True

    def test_upload_file(self):
        file_content = b"Test file content"
        mock_file = MagicMock()
        mock_file.filename = 'test_file.txt'
        mock_file.read.return_value = file_content

        response = self.app.post('/upload-file', data={'file': mock_file})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'content': "File uploaded!"})

    

    def test_undefined_method(self):
        response = self.app.post('/undefined-method', data={'message': 'When was Dajana born?'})

        self.assertEqual(response.status_code, 404)

if __name__ == '__main__':
    unittest.main()
