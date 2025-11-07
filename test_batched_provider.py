from src.translation_agency.llm.batched_session_provider import BatchedSessionProvider

print('=== TESTING BATCHED SESSION PROVIDER ===')

provider = BatchedSessionProvider(headless=True, remove_cache=True, debug=False)

# Test starting session
print('Starting session...')
if provider.start_session():
    print('✓ Session started successfully')

    # Test batch execution with simple prompts
    prompts = [
        "Translate 'Hello world' to Spanish",
        "Now translate 'Goodbye world' to Spanish",
        "Finally, translate 'Thank you' to Spanish"
    ]

    print(f'Executing {len(prompts)} prompts in batch...')
    responses = provider.execute_batch(prompts)

    print('✓ Batch execution completed')
    for i, response in enumerate(responses, 1):
        print(f'Response {i}: {response[:100]}...')

    # End session
    provider.end_session()
    print('✓ Session ended')
else:
    print('✗ Failed to start session')