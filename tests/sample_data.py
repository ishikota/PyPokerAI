round_state1 = {
  'round_count': 3,
  'dealer_btn': 2,
  'next_player': 2,
  'small_blind_pos': 0,
  'big_blind_pos': 1,
  'small_blind_amount': 10,
  'street': 'flop',
  'community_card': ['HK', 'CQ', 'DQ'],
  'seats': [
    {'stack': 80, 'state': 'participating', 'name': 'p1', 'uuid': 'zjwhieqjlowtoogemqrjjo'},
    {'stack': 0, 'state': 'allin', 'name': 'p2', 'uuid': 'xgbpujiwtcccyicvfqffgy'},
    {'stack': 120, 'state': 'participating', 'name': 'p3', 'uuid': 'pnqfqsvgwkegkuwnzucvxw'}
  ],
  'pot': {
    'main': {'amount': 100}, 
    'side': [{'amount': 0, 'eligibles': ['zjwhieqjlowtoogemqrjjo', 'xgbpujiwtcccyicvfqffgy']}]
   },
  'action_histories': {
    'preflop': [
      {'action': 'ANTE', 'amount': 5, 'uuid': 'zjwhieqjlowtoogemqrjjo'},
      {'action': 'ANTE', 'amount': 5, 'uuid': 'xgbpujiwtcccyicvfqffgy'},
      {'action': 'ANTE', 'amount': 5, 'uuid': 'pnqfqsvgwkegkuwnzucvxw'},
      {'action': 'SMALLBLIND', 'amount': 10, 'add_amount': 10, 'uuid': 'zjwhieqjlowtoogemqrjjo'},
      {'action': 'BIGBLIND', 'amount': 20, 'add_amount': 10, 'uuid': 'xgbpujiwtcccyicvfqffgy'},
      {'action': 'RAISE', 'amount': 30, 'add_amount': 10, 'paid': 25, 'uuid': 'pnqfqsvgwkegkuwnzucvxw'},
      {'action': 'CALL', 'amount': 30, 'uuid': 'zjwhieqjlowtoogemqrjjo', 'paid': 20},
      {'action': 'CALL', 'amount': 30, 'uuid': 'xgbpujiwtcccyicvfqffgy', 'paid': 10}
    ],
    'flop': [
      {'action': 'CALL', 'amount': 0, 'uuid': 'zjwhieqjlowtoogemqrjjo', 'paid': 0}
     ]
  }
 }
 
round_state2 = {
  'round_count': 2,
  'street': 'turn',
  'dealer_btn': 1,
  'small_blind_pos': 2,
  'big_blind_pos': 0,
  'next_player': 0,
  'small_blind_amount': 5,
  'community_card': ['CT', 'H9', 'S3', 'CA'],
  'pot': {
    'main': {'amount': 150},
    'side': [{'amount': 10, 'eligibles': ['zjwhieqjlowtoogemqrjjo']}]
  },
  'seats': [
    {'stack': 95, 'state': 'participating', 'name': 'p1', 'uuid': 'zjwhieqjlowtoogemqrjjo'},
    {'stack': 45, 'state': 'participating', 'name': 'p2', 'uuid': 'xgbpujiwtcccyicvfqffgy'},
    {'stack': 0, 'state': 'allin', 'name': 'p3', 'uuid': 'pnqfqsvgwkegkuwnzucvxw'}
   ],
  'action_histories': {
    'preflop': [
      {'action': 'SMALLBLIND', 'amount': 5, 'add_amount': 5, 'uuid': 'pnqfqsvgwkegkuwnzucvxw'},
      {'action': 'BIGBLIND', 'amount': 10, 'add_amount': 5, 'uuid': 'zjwhieqjlowtoogemqrjjo'},
      {'action': 'CALL', 'amount': 10, 'uuid': 'xgbpujiwtcccyicvfqffgy', 'paid': 10},
      {'action': 'CALL', 'amount': 10, 'uuid': 'pnqfqsvgwkegkuwnzucvxw', 'paid': 5},
      {'action': 'CALL', 'amount': 10, 'uuid': 'zjwhieqjlowtoogemqrjjo', 'paid': 0}
     ],
     'flop': [
       {'action': 'RAISE', 'amount': 40, 'add_amount': 40, 'paid': 40, 'uuid': 'pnqfqsvgwkegkuwnzucvxw'},
       {'action': 'CALL', 'amount': 40, 'uuid': 'zjwhieqjlowtoogemqrjjo', 'paid': 40},
       {'action': 'CALL', 'amount': 40, 'uuid': 'xgbpujiwtcccyicvfqffgy', 'paid': 40}
     ],
    'turn': [
      {'action': 'RAISE', 'amount': 10, 'add_amount': 10, 'paid': 10, 'uuid': 'zjwhieqjlowtoogemqrjjo'}
    ]
  }
} 