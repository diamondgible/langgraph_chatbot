#!/usr/bin/env python3
"""
Test script for chat_generic.py with the provided input messages.
Tests both the direct function call and the LangChain ChatGeneric class.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'langchain_generic'))

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_generic.chat_generic import generate_response_with_chat_completion, ChatGeneric

def test_direct_function_call():
    """Test the direct function call with the provided messages."""
    print("=" * 60)
    print("TESTING DIRECT FUNCTION CALL")
    print("=" * 60)
    
    # Test messages as provided
    test_messages = [
        {"role": "system", "content": "You are a linguistic expert. Analyse the given text's sentiment and infer if it's positive, negative or neutral."},
        {"role": "user", "content": "I am going out for dinner tomorrow night"}
    ]
    
    print("Input messages:")
    for i, msg in enumerate(test_messages):
        print(f"  {i+1}. {msg['role']}: {msg['content']}")
    
    try:
        print("\nCalling generate_response_with_chat_completion...")
        response, input_tokens, output_tokens = generate_response_with_chat_completion(test_messages, 
        temperature=0.7, 
        model="Meta-Llama-3.1-8B-Instruct", 
        max_tokens=225)
        
        print(f"\nResponse: {response}")
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print("[SUCCESS] Direct function call successful!")
        
    except Exception as e:
        print(f"[ERROR] Direct function call failed: {e}")
        return False
    
    return True

def test_langchain_chatgeneric():
    """Test the LangChain ChatGeneric class with the provided messages."""
    print("\n" + "=" * 60)
    print("TESTING LANGCHAIN CHATGENERIC CLASS")
    print("=" * 60)
    
    # Convert to LangChain message format
    test_messages = [
        SystemMessage(content="You are a linguistic expert. Analyse the given text's sentiment and infer if it's positive, negative or neutral."),
        HumanMessage(content="I am going out for dinner tomorrow night")
    ]
    
    print("Input messages (LangChain format):")
    for i, msg in enumerate(test_messages):
        print(f"  {i+1}. {type(msg).__name__}: {msg.content}")
    
    try:
        print("\nCreating ChatGeneric instance...")
        chat_model = ChatGeneric(model="Meta-Llama-3.1-8B-Instruct", max_tokens=225)
        
        print("Calling chat_model.invoke...")
        result = chat_model.invoke(test_messages)
        
        print(f"\nResponse: {result.content}")
        
        print(f"Result: {result}")
        
        # Check if generation info is available
        if hasattr(result, 'generation_info') and result.generation_info:
            print(f"Generation info: {result.generation_info}")
        
        print("[SUCCESS] LangChain ChatGeneric class successful!")
        
    except Exception as e:
        print(f"[ERROR] LangChain ChatGeneric class failed: {e}")
        return False
    
    return True

def test_langchain_generate():
    """Test the _generate method directly."""
    print("\n" + "=" * 60)
    print("TESTING LANGCHAIN _GENERATE METHOD")
    print("=" * 60)
    
    # Convert to LangChain message format
    test_messages = [
        SystemMessage(content="You are a linguistic expert. Analyse the given text's sentiment and infer if it's positive, negative or neutral."),
        HumanMessage(content="I am going out for dinner tomorrow night")
    ]
    
    try:
        print("Creating ChatGeneric instance...")
        chat_model = ChatGeneric(model="Meta-Llama-3.1-8B-Instruct", temperature=0.7, max_tokens=225)
        
        print("Calling chat_model._generate...")
        result = chat_model._generate(test_messages)
        
        print(f"\nResponse: {result.generations[0].message.content}")
        
        # Check generation info
        if result.generations[0].generation_info:
            print(f"Generation info: {result.generations[0].generation_info}")
        
        print("[SUCCESS] LangChain _generate method successful!")
        
    except Exception as e:
        print(f"[ERROR] LangChain _generate method failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Testing chat_generic.py with sentiment analysis input")
    print("Expected: The model should analyze the sentiment of 'I am going out for dinner tomorrow night'")
    print("Expected sentiment: Neutral (no specific sentiment)")
    
    # Check if API key is available
    if not os.getenv("LLM_API_KEY"):
        print("[ERROR] LLM_API_KEY environment variable not found!")
        print("Please set your LLM_API_KEY in a .env file or environment variable.")
        return

    # Check if base URL is available
    if not os.getenv("LLM_BASE_URL"):
        print("[ERROR] LLM_BASE_URL environment variable not found!")
        print("Please set your LLM_BASE_URL in a .env file or environment variable.")
        return
    
    success_count = 0
    total_tests = 3
    
    # Run tests
    if test_direct_function_call():
        success_count += 1
    
    if test_langchain_chatgeneric():
        success_count += 1
    
    if test_langchain_generate():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("[SUCCESS] All tests passed!")
    else:
        print("[WARNING] Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
