import datasets
from datasets import load_dataset
import random
import numpy as np
import csv
import sys
import os
import pandas as pd

datasets.logging.set_verbosity(datasets.logging.ERROR)

clinc_label_to_explanation={
    "restaurant reviews": "restaurant reviews: Provide evaluations and comments about restaurants.",
    "nutrition info": "nutrition info: Offer nutritional information about food items.",
    "account blocked": "account blocked: Notify the user that their account is blocked or frozen.",
    "oil change how": "oil change how: Provide guidance on how to change car oil.",
    "time": "time: Offer current time or time in a specific time zone.",
    "weather": "weather: Provide current or location-specific weather conditions.",
    "redeem rewards": "redeem rewards: Assist users in redeeming points or rewards.",
    "interest rate": "interest rate: Provide information about interest rates, possibly for loans or deposits.",
    "gas type": "gas type: Provide information about different types of gasoline or fuel.",
    "accept reservations": "accept reservations: Confirm whether a restaurant or venue accepts reservations.",
    "smart home": "smart home: Control smart home devices or provide related information.",
    "user name": "user name: Retrieve or update the user's username.",
    "report lost card": "report lost card: Report a user's lost credit or debit card.",
    "repeat": "repeat: Repeat the last command or information.",
    "whisper mode": "whisper mode: Switch to a silent or private mode.",
    "what are your hobbies": "what are your hobbies: Inquire about the chatbot's hobbies.",
    "order": "order: Place an order or purchase items.",
    "jump start": "jump start: Provide methods or services for jump-starting a car.",
    "schedule meeting": "schedule meeting: Arrange a meeting or schedule a time.",
    "meeting schedule": "meeting schedule: Provide meeting schedules or agendas.",
    "freeze account": "freeze account: Freeze the user's account.",
    "what song": "what song: Identify the currently playing song or provide music recommendations.",
    "meaning of life": "meaning of life: Answer questions about the meaning of life.",
    "restaurant reservation": "restaurant reservation: Reserve seats or tables at a restaurant.",
    "traffic": "traffic: Provide traffic conditions or route suggestions.",
    "make call": "make call: Make a call to a specific contact or number.",
    "text": "text: Send text messages or handle text messages.",
    "bill balance": "bill balance: Inquire about bill balances or account balances.",
    "improve credit score": "improve credit score: Provide advice or methods to improve credit scores.",
    "change language": "change language: Change language settings or provide multilingual support.",
    "no": "no: Respond to negative questions or refuse requests.",
    "measurement conversion": "measurement conversion: Provide unit conversions or conversions between measurement units.",
    "timer": "timer: Set timers or reminders.",
    "flip coin": "flip coin: Flip a coin to decide or provide random choices.",
    "do you have pets": "do you have pets: Inquire if the chatbot has pets.",
    "balance": "balance: Check account balances or balances of other resources.",
    "tell joke": "tell joke: Tell jokes or provide humor.",
    "last maintenance": "last maintenance: Provide information about the last maintenance or service.",
    "exchange rate": "exchange rate: Provide currency exchange rates or conversion rates.",
    "uber": "uber: Request Uber services or provide related information.",
    "car rental": "car rental: Rent a car or inquire about car rental services.",
    "credit limit": "credit limit: Inquire about or change credit limits.",
    "shopping list": "shopping list: Create, view, or edit shopping lists.",
    "expiration date": "expiration date: Inquire about or provide expiration date information.",
    "routing": "routing: Inquire about bank account routing numbers or routing information.",
    "meal suggestion": "meal suggestion: Provide meal suggestions or recipe recommendations.",
    "tire change": "tire change: Change car tires or provide related services.",
    "todo list": "todo list: Create, manage, or view to-do lists.",
    "card declined": "card declined: Notify the user of declined credit cards.",
    "rewards balance": "rewards balance: Check points or rewards balances.",
    "change accent": "change accent: Change voice or accent settings.",
    "vaccines": "vaccines: Provide vaccine information or vaccination recommendations.",
    "reminder update": "reminder update: Update or manage reminders.",
    "food last": "food last: Inquire about the last purchased food or ingredients.",
    "change ai name": "change ai name: Change the chatbot's name.",
    "bill due": "bill due: Inquire about or remind of bill due dates.",
    "who do you work for": "who do you work for: Inquire about the manufacturer of the chatbot or the company behind it.",
    "share location": "share location: Share current location or location information.",
    "international visa": "international visa: Provide international visa information or application advice.",
    "calendar": "calendar: Access or manage calendar events.",
    "translate": "translate: Provide translation services or language translations.",
    "carry on": "carry on: Continue with previous conversation or tasks.",
    "book flight": "book flight: Book flights or inquire about airline tickets.",
    "insurance change": "insurance change: Change insurance policies or provide insurance services.",
    "todo list update": "todo list update: Update to-do lists or tasks.",
    "timezone": "timezone: Inquire about or change timezone settings.",
    "cancel reservation": "cancel reservation: Cancel reservations or appointments.",
    "transactions": "transactions: Inquire about or manage transaction records.",
    "credit score": "credit score: Check credit scores or improve credit scores.",
    "report fraud": "report fraud: Report fraudulent activities or provide fraud protection.",
    "spending history": "spending history: Inquire about or view spending history.",
    "directions": "directions: Provide navigation or map directions.",
    "spelling": "spelling: Spell check or provide spelling suggestions.",
    "insurance": "insurance: Inquire about or purchase insurance.",
    "what is your name": "what is your name: Inquire about the chatbot's name.",
    "reminder": "reminder: Create, manage, or remind of events.",
    "where are you from": "where are you from: Inquire about the chatbot's origin or source.",
    "distance": "distance: Calculate distance or provide explanations.",
    "payday": "payday: Provide information or reminders about payday.",
    "flight status": "flight status: Check the status of a flight or provide flight information.",
    "find phone": "find phone: Assist in locating a lost or misplaced phone.",
    "greeting": "greeting: Greet the user or initiate a conversation.",
    "alarm": "alarm: Set or manage alarms or reminders.",
    "order status": "order status: Inquire about the status of an order or delivery.",
    "confirm reservation": "confirm reservation: Confirm a reservation or booking.",
    "cook time": "cook time: Provide cooking time for recipes or dishes.",
    "damaged card": "damaged card: Report a damaged credit or debit card.",
    "reset settings": "reset settings: Reset user settings or preferences.",
    "pin change": "pin change: Change or update PIN numbers.",
    "replacement card duration": "replacement card duration: Provide information on the duration for receiving a replacement card.",
    "new card": "new card: Order or activate a new credit or debit card.",
    "roll dice": "roll dice: Simulate rolling dice or provide random numbers.",
    "income": "income: Inquire about or provide information on income.",
    "taxes": "taxes: Provide information or assistance regarding taxes.",
    "date": "date: Provide the current date or assist with date-related queries.",
    "who made you": "who made you: Provide information about the creator or developer of the chatbot.",
    "pto request": "pto request: Request time off or vacation time.",
    "tire pressure": "tire pressure: Check or adjust tire pressure.",
    "how old are you": "how old are you: Inquire about the age of the chatbot.",
    "rollover 401k": "rollover 401k: Provide information or assistance regarding 401k rollovers.",
    "pto request status": "pto request status: Check the status of a time off or vacation request.",
    "how busy": "how busy: Inquire about how busy a location or establishment is.",
    "application status": "application status: Check the status of an application or request.",
    "recipe": "recipe: Provide recipes or cooking instructions.",
    "calendar update": "calendar update: Update calendar events or schedules.",
    "play music": "play music: Play music or provide music recommendations.",
    "yes": "yes: Respond affirmatively to questions or requests.",
    "direct deposit": "direct deposit: Set up or manage direct deposit for payments.",
    "credit limit change": "credit limit change: Request or implement changes to credit limits.",
    "gas": "gas: Provide information about gas stations or fuel prices.",
    "pay bill": "pay bill: Pay bills or manage bill payments.",
    "ingredients list": "ingredients list: Provide a list of ingredients for recipes or dishes.",
    "lost luggage": "lost luggage: Report lost or missing luggage.",
    "goodbye": "goodbye: Bid farewell to the user or end the conversation.",
    "what can i ask you": "what can i ask you: Provide guidance on what questions can be asked.",
    "book hotel": "book hotel: Reserve hotel accommodations or inquire about hotel bookings.",
    "are you a bot": "are you a bot: Confirm whether the entity is a chatbot or not.",
    "next song": "next song: Skip to the next song in a playlist or queue.",
    "change speed": "change speed: Adjust playback speed for media or content.",
    "plug type": "plug type: Provide information about plug types or standards.",
    "maybe": "maybe: Indicate uncertainty or ambiguity in response to questions or requests.",
    "w2": "w2: Provide or request W2 tax forms.",
    "oil change when": "oil change when: Determine when an oil change is due for a vehicle.",
    "thank you": "thank you: Express gratitude or appreciation.",
    "shopping list update": "shopping list update: Update or modify items on a shopping list.",
    "pto balance": "pto balance: Inquire about remaining paid time off balance.",
    "order checks": "order checks: Place an order for checks or checkbooks.",
    "travel alert": "travel alert: Provide alerts or notifications regarding travel advisories.",
    "fun fact": "fun fact: Share interesting or fun facts.",
    "sync device": "sync device: Synchronize or connect devices.",
    "schedule maintenance": "schedule maintenance: Arrange or schedule maintenance tasks.",
    "apr": "apr: Provide information on Annual Percentage Rates (APR).",
    "transfer": "transfer: Transfer funds or assets between accounts.",
    "ingredient substitution": "ingredient substitution: Provide substitutes for cooking ingredients.",
    "calories": "calories: Provide calorie information for food items.",
    "current location": "current location: Provide or determine the current location.",
    "international fees": "international fees: Provide information on international transaction fees.",
    "calculator": "calculator: Perform calculations or provide a calculator tool.",
    "definition": "definition: Provide definitions for words or terms.",
    "next holiday": "next holiday: Provide information about the next upcoming holiday.",
    "update playlist": "update playlist: Add or remove songs from a playlist.",
    "mpg": "mpg: Calculate or provide information on miles per gallon (MPG).",
    "min payment": "min payment: Determine the minimum payment due on an account.",
    "change user name": "change user name: Modify or update user names.",
    "restaurant suggestion": "restaurant suggestion: Provide recommendations for restaurants.",
    "travel notification": "travel notification: Provide notifications or alerts related to travel plans.",
    "cancel": "cancel: Cancel an action, service, or subscription.",
    "pto used": "pto used: Record or track used paid time off.",
    "travel suggestion": "travel suggestion: Offer suggestions or recommendations for travel destinations.",
    "change volume": "change volume: Adjust or modify the volume level."
}

bank_label_to_explanation={
    "Refund not showing up": "Refund not showing up: Users inquire about the status of a refund that hasn't appeared in their account.",
    "activate my card": "activate my card: Users seek guidance on how to activate their card for use.",
    "age limit": "age limit: Users inquire about any age restrictions or requirements associated with a service or product.",
    "apple pay or google pay": "apple pay or google pay: Users want to know if the service supports payment methods such as Apple Pay or Google Pay.",
    "atm support": "atm support: Users inquire whether the service provides support for Automated Teller Machines (ATMs) for functions like withdrawals or balance inquiries.",
    "automatic top up": "automatic top up: Users want to know if their account can be set to automatically add funds when the balance falls below a certain threshold.",
    "balance not updated after bank transfer": "balance not updated after bank transfer: Users report that their account balance hasn't been updated after making a bank transfer.",
    "balance not updated after cheque or cash deposit": "balance not updated after cheque or cash deposit: Users report that their account balance hasn't been updated after depositing a cheque or cash.",
    "beneficiary not allowed": "beneficiary not allowed: Users are informed that the recipient they've tried to send money to is not permitted by the service for some reason.",
    "cancel transfer": "cancel transfer: Users want to cancel a pending transfer they've initiated.",
    "card about to expire": "card about to expire: Users want information about their card that is nearing its expiration date.",
    "card acceptance": "card acceptance: Users inquire about the acceptance of their card at specific merchants or locations.",
    "card arrival": "card arrival: Users want to know when they can expect to receive their physical card in the mail.",
    "card delivery estimate": "card delivery estimate: Users seek an estimated delivery time for their physical card.",
    "card linking": "card linking: Users seek guidance on how to link their card to their account or another service.",
    "card not working": "card not working: Users report that their card is not functioning properly and seek assistance.",
    "card payment fee charged": "card payment fee charged: Users notice a fee charged for a card payment and seek clarification.",
    "card payment not recognised": "card payment not recognised: Users do not recognize a card payment transaction and seek clarification or resolution.",
    "card payment wrong exchange rate": "card payment wrong exchange rate: Users notice an incorrect exchange rate applied to a card payment and seek resolution.",
    "card swallowed": "card swallowed: Users report that their card was swallowed by an ATM and seek assistance.",
    "cash withdrawal charge": "cash withdrawal charge: Users notice a fee charged for a cash withdrawal and seek clarification.",
    "cash withdrawal not recognised": "cash withdrawal not recognised: Users do not recognize a cash withdrawal transaction and seek clarification or resolution.",
    "change pin": "change pin: Users want to change the Personal Identification Number (PIN) associated with their card.",
    "compromised card": "compromised card: Users suspect their card information has been compromised and seek assistance.",
    "contactless not working": "contactless not working: Users report that the contactless feature of their card is not functioning properly and seek assistance.",
    "country support": "country support: Users inquire about the availability of services or support in specific countries.",
    "declined card payment": "declined card payment: Users report that their card payment was declined and seek assistance.",
    "declined cash withdrawal": "declined cash withdrawal: Users report that their cash withdrawal was declined and seek assistance.",
    "declined transfer": "declined transfer: Users report that a transfer they initiated was declined and seek assistance.",
    "direct debit payment not recognised": "direct debit payment not recognised: Users do not recognize a direct debit payment transaction and seek clarification or resolution.",
    "disposable card limits": "disposable card limits: Users inquire about limits associated with disposable or temporary cards.",
    "edit personal details": "edit personal details: Users want to edit or update personal information associated with their account.",
    "exchange charge": "exchange charge: Users notice a charge associated with currency exchange and seek clarification.",
    "exchange rate": "exchange rate: Users inquire about the current exchange rate for currency conversion.",
    "exchange via app": "exchange via app: Users want to exchange currency within the app or platform.",
    "extra charge on statement": "extra charge on statement: Users notice an additional charge on their statement and seek clarification.",
    "failed transfer": "failed transfer: Users report that a transfer they initiated has failed and seek assistance.",
    "fiat currency support": "fiat currency support: Users inquire about support for traditional fiat currencies within the service.",
    "get disposable virtual card": "get disposable virtual card: Users want to obtain a disposable or temporary virtual card.",
    "get physical card": "get physical card: Users want to obtain a physical card.",
    "getting spare card": "getting spare card: Users want to obtain a spare or additional card.",
    "getting virtual card": "getting virtual card: Users want to obtain a virtual card.",
    "lost or stolen card": "lost or stolen card: Users report their card as lost or stolen and seek assistance.",
    "lost or stolen phone": "lost or stolen phone: Users report their phone as lost or stolen and seek assistance.",
    "order physical card": "order physical card: Users want to order a physical card.",
    "passcode forgotten": "passcode forgotten: Users have forgotten their passcode and seek assistance in resetting it.",
    "pending card payment": "pending card payment: Users inquire about the status of a pending card payment transaction.",
    "pending cash withdrawal": "pending cash withdrawal: Users inquire about the status of a pending cash withdrawal transaction.",
    "pending top up": "pending top up: Users inquire about the status of a pending account top-up transaction.",
    "pending transfer": "pending transfer: Users inquire about the status of a pending transfer transaction.",
    "pin blocked": "pin blocked: Users report that their PIN has been blocked and seek assistance.",
    "receiving money": "receiving money: Users want to know how to receive money into their account.",
    "request refund": "request refund: Users want to request a refund for a transaction or service.",
    "reverted card payment?": "reverted card payment?: Users inquire about a transaction that has been reverted or reversed.",
    "supported cards and currencies": "supported cards and currencies: Users inquire about the types of cards and currencies supported by the service.",
    "terminate account": "terminate account: Users want to terminate or close their account.",
    "top up by bank transfer charge": "top up by bank transfer charge: Users notice a charge associated with topping up their account via bank transfer and seek clarification.",
    "top up by card charge": "top up by card charge: Users notice a charge associated with topping up their account via card and seek clarification.",
    "top up by cash or cheque": "top up by cash or cheque: Users want to know if they can top up their account using cash or cheque.",
    "top up failed": "top up failed: Users report that a top-up transaction has failed and seek assistance.",
    "top up limits": "top up limits: Users inquire about any limits associated with topping up their account.",
    "top up reverted": "top up reverted: Users notice that a top-up transaction has been reverted or reversed and seek clarification.",
    "topping up by card": "topping up by card: Users want to know how to top up their account using a card.",
    "transaction charged twice": "transaction charged twice: Users notice that a transaction has been charged twice and seek clarification or resolution.",
    "transfer fee charged": "transfer fee charged: Users notice a fee charged for a transfer and seek clarification.",
    "transfer into account": "transfer into account: Users want to know how to transfer money into their account from another source.",
    "transfer not received by recipient": "transfer not received by recipient: Users report that a transfer they initiated has not been received by the intended recipient and seek assistance.",
    "transfer timing": "transfer timing: Users inquire about the timing or speed of transfers.",
    "unable to verify identity": "unable to verify identity: Users report that they are unable to verify their identity and seek assistance.",
    "verify my identity": "verify my identity: Users want to verify their identity for security or compliance purposes.",
    "verify source of funds": "verify source of funds: Users are asked to verify the source of funds for a transaction or account.",
    "verify top up": "verify top up: Users are asked to verify a top-up transaction for security or compliance purposes.",
    "virtual card not working": "virtual card not working: Users report that their virtual card is not functioning properly and seek assistance.",
    "visa or mastercard": "visa or mastercard: Users inquire about whether the service supports Visa or Mastercard.",
    "why verify identity": "why verify identity: Users seek an explanation for why they are being asked to verify their identity.",
    "wrong amount of cash received": "wrong amount of cash received: Users report that they received an incorrect amount of cash and seek assistance.",
    "wrong exchange rate for cash withdrawal": "wrong exchange rate for cash withdrawal: Users notice an incorrect exchange rate applied to a cash withdrawal and seek resolution."
}

task_to_keys = {
    'clinc150': ("text", None),
    'bank': ("text", None),
    "stackoverflow": ("text", None)
}


def load(task_name, tokenizer, shot=0, max_seq_length=256, is_id=False, dir=None, known_cls_ratio=None):
    sentence1_key, sentence2_key = task_to_keys[task_name]
    print("Loading {}".format(task_name))

    if task_name == 'clinc150':
        datasets, num_labels = load_clinc(is_id, shot=shot, dir=dir, known_cls_ratio=known_cls_ratio)
    elif task_name == 'ROSTD':
        datasets, num_labels = load_clinc(is_id, shot=shot)
    elif task_name == 'bank':
        datasets, num_labels = load_uood(is_id, shot=shot, dir=dir, known_cls_ratio=known_cls_ratio)
    elif task_name == 'stackoverflow':
        datasets, num_labels = load_uood(is_id, shot=shot, dir=dir, known_cls_ratio=known_cls_ratio)
    else:
        print("task is not supported")

    def preprocess_function(examples):
        inputs = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key] + " " + examples[sentence2_key],)
        )
        result = tokenizer(*inputs, max_length=max_seq_length, truncation=True)
        result["labels"] = examples["label"] if 'label' in examples else 0
        return result

    train_dataset = list(map(preprocess_function, datasets['train'])) if 'train' in datasets and is_id else None
    dev_dataset = list(map(preprocess_function, datasets['validation'])) if 'validation' in datasets and is_id else None
    test_dataset = list(map(preprocess_function, datasets['test'])) if 'test' in datasets else None
    print(test_dataset[0])
    print(test_dataset[1])
    id_dataset = list(map(preprocess_function, datasets['id'])) if 'id' in datasets else None
    return train_dataset, dev_dataset, test_dataset,id_dataset, num_labels


def load_clinc(is_id, shot=None, dir=None, known_cls_ratio=None):
    # label_list = get_labels(dir)
    # label_map = {}
    # for i, label in enumerate(label_list):
    #     label_map[label] = i
    all_label_list_pos = get_labels(dir)
    n_known_cls = round(len(all_label_list_pos) * known_cls_ratio)
    known_label_list = list(
        np.random.choice(np.array(all_label_list_pos), n_known_cls, replace=False))
    print(known_label_list)
    ood_labels = list(set(all_label_list_pos) - set(known_label_list))
    label_map = {}
    for i, label in enumerate(known_label_list):
        label_map[label] = i
    label_map["oos"] = n_known_cls

    train_dataset = _create_examples(
        _read_tsv(os.path.join(dir, "train.tsv")), label_map, known_label_list)
    dev_dataset = _create_examples(
        _read_tsv(os.path.join(dir, "dev.tsv")), label_map, known_label_list)
    test_dataset = _create_examples(
        _read_tsv(os.path.join(dir, "test.tsv")), label_map, known_label_list)


    shots = get_shots(shot, train_dataset, "clinc150")
    train_dataset = train_select_few_shot(shots, train_dataset, "clinc150")
    dev_dataset = select_few_shot(shots, dev_dataset, "clinc150")


    ood_dataset = _get_ood(
        _read_tsv(os.path.join(dir, "test.tsv")), ood_labels + ["oos"], label_map)
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset+ood_dataset, "id": test_dataset}
    # datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset, "ood": ood_dataset}
    return datasets, n_known_cls


def load_uood(is_id, shot=None, dir=None, known_cls_ratio=None):
    all_label_list_pos = get_labels(dir)
    n_known_cls = round(len(all_label_list_pos) * known_cls_ratio)
    known_label_list = list(
        np.random.choice(np.array(all_label_list_pos), n_known_cls, replace=False))

    ood_labels = list(set(all_label_list_pos) - set(known_label_list))
    label_map = {}
    for i, label in enumerate(known_label_list):
        label_map[label] = i
    label_map["oos"] = n_known_cls
    print(label_map)
    train_dataset = _create_examples(
        _read_tsv(os.path.join(dir, "train.tsv")), label_map, known_label_list)
    dev_dataset = _create_examples(
        _read_tsv(os.path.join(dir, "dev.tsv")), label_map, known_label_list)
    test_dataset = _create_examples(
        _read_tsv(os.path.join(dir, "test.tsv")), label_map, known_label_list)

    shots = get_shots(shot, train_dataset, "bank")
    train_dataset = train_select_few_shot(shots, train_dataset, "bank")
    dev_dataset = select_few_shot(shots, dev_dataset, "bank")

    ood_dataset = _get_ood(
        _read_tsv(os.path.join(dir, "test.tsv")), ood_labels, label_map)
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset+ood_dataset, "id": test_dataset}
    print(datasets['test'][3])
    return datasets, n_known_cls


def load_ROSTD(is_id, shot=100, data_dir="/data1/liming/ROSTD"):
    label_list = get_labels(data_dir)
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
    ood_list = ['oos']

    if is_id:
        train_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "train.tsv")), label_map, label_list)
        dev_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "dev.tsv")), label_map, label_list)
        test_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "test.tsv")), label_map, label_list)
        train_dataset = select_few_shot(shot, train_dataset, "clinc150")
        dev_dataset = select_few_shot(shot, dev_dataset, "clinc150")
        datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    else:
        test_dataset = _get_ood(
            _read_tsv(os.path.join(data_dir, "test.tsv")), ood_list)
        datasets = {'test': test_dataset}
    return datasets

def get_shots(shot, trainset, task_name):
    # examples = []
    few_examples = []
    sentence1_key, sentence2_key = task_to_keys[task_name]
    from collections import defaultdict
    sorted_examples = defaultdict(list)

    for example in trainset:
        # if example.label in self.known_label_list and np.random.uniform(0, 1) <= args.labeled_ratio:
        #     examples.append(example)
        sorted_examples[example["label"]] = sorted_examples[example["label"]] + [example[sentence1_key]]
    k, v = list(sorted_examples.items())[0]

    return round(len(v) * shot)
    # for k, v in sorted_examples.items():
    #     arr = np.array(v)
    #     shot_n = round(len(arr) * shot)
    #     np.random.shuffle(arr)
    #     for elems in arr[:shot_n]:
    #         few_examples.append({sentence1_key: elems, 'label': k})
    #
    # return few_examples

def train_select_few_shot(shot, trainset, task_name):
    # examples = []
    few_examples = []
    sentence1_key, sentence2_key = task_to_keys[task_name]
    from collections import defaultdict
    sorted_examples = defaultdict(list)

    for example in trainset:
        # if example.label in self.known_label_list and np.random.uniform(0, 1) <= args.labeled_ratio:
        #     examples.append(example)
        sorted_examples[example["label"]] = sorted_examples[example["label"]] + [(example[sentence1_key], example['label_text'])]
    for k, v in sorted_examples.items():
        arr = np.array(v)
        # shot_n = round(len(arr) * shot)
        np.random.shuffle(arr)
        for elems in arr[:shot]:
            # few_examples.append({sentence1_key: elems, 'label': k})
            few_examples.append({sentence1_key: elems[0], 'label': k, 'label_text': elems[1]})
            # combined_text = f"{elems[0]} [SEP] {elems[1]}"
            # combined_text = f"{elems[0]} {elems[1]}"
            combined_text = f"{elems[0]} [SEP] {bank_label_to_explanation[elems[1]]}"
            # combined_text = f"{elems[0]} [SEP] The intent of this utterance is: {elems[1]}"
            # combined_text = f"{elems[0]} The intent of this utterance is: {elems[1]}"
            # if not any(elem[sentence1_key] == elems[1] for elem in few_examples):
            few_examples.append({sentence1_key: combined_text, 'label': k, 'label_text': elems[1]})
    print(few_examples[0])
    print(few_examples[1])
    return few_examples

def select_few_shot(shot, trainset, task_name):
    # examples = []
    few_examples = []
    sentence1_key, sentence2_key = task_to_keys[task_name]
    from collections import defaultdict
    sorted_examples = defaultdict(list)

    for example in trainset:
        # if example.label in self.known_label_list and np.random.uniform(0, 1) <= args.labeled_ratio:
        #     examples.append(example)
        sorted_examples[example["label"]] = sorted_examples[example["label"]] + [(example[sentence1_key], example['label_text'])]
    for k, v in sorted_examples.items():
        arr = np.array(v)
        # shot_n = round(len(arr) * shot)
        np.random.shuffle(arr)
        for elems in arr[:shot]:
            # few_examples.append({sentence1_key: elems, 'label': k})
            few_examples.append({sentence1_key: elems[0], 'label': k, 'label_text': elems[1]})
            # combined_text = f"{elems[0]} [SEP] {elems[1]}"
            # combined_text = f"{elems[0]} [SEP] The intent of this utterance is: {elems[1]}"
            # combined_text = f"{elems[0]} The intent of this utterance is: {elems[1]}"
            # if not any(elem[sentence1_key] == elems[1] for elem in few_examples):
            # few_examples.append({sentence1_key: combined_text, 'label': k, 'label_text': elems[1]})
    return few_examples
def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def _create_examples(lines, label_map, know_labels):
    """Creates examples for the training and dev sets."""
    examples = []

    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        if len(line) != 2:
            continue
        # guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[1]
        if label in know_labels:
            label_text = label.replace("_", " ") if "_" in label else label
            examples.append(
                {'text': text_a, 'label': label_map[label], 'label_text':label_text})
    return examples


def _get_ood(lines, ood_labels, label_map):
    out_examples = []

    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        if len(line) != 2:
            continue
        # guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[1]
        if label in ood_labels:
            out_examples.append(
                {'text': text_a, 'label': label_map["oos"]})

    return out_examples


def get_labels(data_dir):
    """See base class."""
    import pandas as pd
    test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
    labels = np.unique(np.array(test['label']))

    return labels
