import pytest
from transformers import AutoTokenizer
from llm_perf_tools.inference import InferenceTracker


def test_inference_tracker_default_tokenizer():
    class MockClient:
        pass
    
    client = MockClient()
    tracker = InferenceTracker(client)
    
    assert tracker.tokenizer is not None
    
    text = "Hello world, this is a test message"
    token_count = tracker.tokenizer(text)
    assert token_count > 0
    assert isinstance(token_count, int)


def test_inference_tracker_custom_tokenizer():
    class MockClient:
        pass
    
    client = MockClient()
    custom_tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    tokenizer_func = lambda text: len(custom_tokenizer.encode(text))
    tracker = InferenceTracker(client, tokenizer=tokenizer_func)
    
    assert tracker.tokenizer == tokenizer_func
    
    text = "Hello world, this is a test message"
    token_count = tracker.tokenizer(text)
    assert token_count > 0
    assert isinstance(token_count, int)