from clearml.automation import TriggerScheduler

# create the TriggerScheduler object (checking system state every minute)
trigger = TriggerScheduler(pooling_frequency_minutes=1.0)

# Add trigger on dataset creation
trigger.add_dataset_trigger(
    name='Retrain On Dataset',
    # schedule_function=lambda x: print("Hey Mom!"),
    schedule_task_id='1f37a06dca1d4f62bb8bf5303756dd79', # you can also schedule an existing task to be executed
    schedule_queue='default',
    trigger_project='Full Overview',
    trigger_name='Fashion MNIST'
)

# Start the trigger remotely, so an agent will keep track of everything
trigger.start_remotely()