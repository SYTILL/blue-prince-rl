# 미니멀 Blue Prince: 방 목록 (초안)

출처: Blue Prince 위키 (Room Directory, Room shape, Gem cost, 색깔별 분류 페이지). 2026-10 기준.

## 규칙 요약
- 방끼리 다른 점은 **모양(문 방향), 보석 비용, 희귀도, 얻는 걸음·열쇠·보석**뿐이다. 특수 효과는 없다.
- 얻는 자원은 **방을 놓는 순간 한 번** 받는다(다시 들어가도 또 받지 않음). ← 확정 필요
- 모양·희귀도·보석 비용은 **원작 그대로**. 얻는 자원은 원작 효과를 단순화한 값이다.
  - `원작` 표시: 원작에서도 거의 같은 양을 준다.
  - `단순화` 표시: 원작은 퍼즐 보상, 무작위 아이템, 다음 날 효과, 조건부 효과 등이라 숫자로 바꿨다. 값은 조정 대상.
  - `빈 방` 표시: 원작 효과를 숫자로 바꾸기 어려워서 **모양만 남기고 아무것도 주지 않는 방**.
- 음수 = 손해. 빨간 방(Red Room) 일부는 원작에서 손해를 주는 방이라 걸음을 깎는 방으로 바꿨다.

## 모양별 배치 규칙
| 모양 | 문 | 놓을 수 있는 곳 |
|---|---|---|
| 막다른 방 (Dead End) | 1 | 어디든 |
| 일자 (Straight) | 2, 마주 봄 | 구석 불가. 가장자리는 가장자리를 따라 들어갈 때만 |
| ㄱ자 (L) | 2, 꺾임 | 방마다 **꺾는 방향(좌/우)이 고정**. 그 방향이 바깥을 향하면 놓을 수 없음 (아래 참고) |
| T자 (T) | 3 | 구석 불가. 가장자리에서는 막힌 쪽이 바깥 |
| 십자 (4-way) | 4 | 가운데만 |

### ㄱ자 방의 방향
- 방마다 **좌회전** 또는 **우회전**이 정해져 있다. 들어온 문에서 방 안을 바라봤을 때 나머지 문이 왼쪽에 있으면 좌, 오른쪽에 있으면 우.
- 그래서 ㄱ자 방은 **들어온 문만 정해지면 놓이는 모양이 하나로 정해진다**(무작위 회전 없음).
  - 예: 북쪽으로 이동해 방을 열면 들어온 문은 남(S). 좌회전 방은 S·W, 우회전 방은 S·E.
- 나머지 문이 집 바깥을 향하면 그 방은 그 자리에 **뽑기 후보로 나오지 않는다**.
  - 가장자리를 따라 들어갈 때: 안쪽으로 꺾는 방만 가능.
  - 가장자리로 곧장 들어갈 때: 좌·우 모두 가능.
  - 구석: 한쪽만 가능.
- 놓은 뒤에는 그냥 문 두 개짜리 방이다. 두 문 모두 양방향으로 지나다닐 수 있다.

```python
LEFT  = {"N": "W", "W": "S", "S": "E", "E": "N"}   # 이동 방향 → 왼쪽
RIGHT = {v: k for k, v in LEFT.items()}            # 이동 방향 → 오른쪽
OPP   = {"N": "S", "S": "N", "E": "W", "W": "E"}

def l_room_doors(move_dir, turn):                  # turn: "L" 또는 "R"
    return {OPP[move_dir], (LEFT if turn == "L" else RIGHT)[move_dir]}
```

## 방 목록 (99개)
고정 방(시작 Entrance Hall, 목표 Antechamber)은 따로 정의하고 덱에는 넣지 않는다.

### 막다른 방 (29)
| 방 | 희귀도 | 보석 비용 | 걸음 | 열쇠 | 보석 | 근거 |
|---|---|---|---|---|---|---|
| Closet | Commonplace | 0 | 0 | +1 | 0 | 단순화 (무작위 아이템) |
| Storeroom | Commonplace | 0 | 0 | +1 | +1 | 원작 |
| Guest Bedroom | Commonplace | 0 | +10 | 0 | 0 | 원작 (첫 입장 +10) |
| Nursery | Commonplace | 1 | +5 | 0 | 0 | 단순화 (침실을 놓을 때마다 +5) |
| Walk-In Closet | Standard | 1 | 0 | +1 | +1 | 단순화 (아이템, 보석 하나는 확정) |
| Terrace | Standard | 0 | 0 | 0 | 0 | 빈 방 |
| Greenhouse | Standard | 1 | 0 | 0 | +2 | 단순화 (보석 꽃 0–3) |
| Lavatory | Standard | 0 | 0 | 0 | 0 | 빈 방 (빨간 방) |
| Utility Closet | Standard | 0 | 0 | 0 | 0 | 빈 방 |
| Coat Check | Standard | 0 | 0 | 0 | 0 | 빈 방 |
| Solarium | Standard | 1 | 0 | 0 | 0 | 빈 방 |
| Dormitory | Standard | 0 | 0 | 0 | 0 | 빈 방 |
| Planetarium | Standard | 0 | 0 | 0 | 0 | 빈 방 |
| Garage | Unusual | 1 | 0 | +3 | 0 | 원작 |
| Wine Cellar | Unusual | 0 | 0 | 0 | +3 | 원작 |
| Study | Unusual | 0 | 0 | 0 | +1 | 원작 (금고 보석 1) |
| Servant's Quarters | Unusual | 1 | 0 | +2 | 0 | 단순화 (침실 수만큼 열쇠) |
| Sauna | Unusual | 0 | 0 | 0 | 0 | 빈 방 |
| Mail Room | Unusual | 0 | 0 | 0 | 0 | 빈 방 |
| Bunk Room | Unusual | 0 | 0 | 0 | 0 | 빈 방 |
| Locksmith | Unusual | 1 | 0 | 0 | 0 | 빈 방 |
| Clock Tower | Unusual | 1 | 0 | 0 | 0 | 빈 방 |
| Master Bedroom | Rare | 2 | +10 | 0 | 0 | 단순화 (다음 날 +10 → 즉시) |
| Attic | Rare | 3 | 0 | +2 | +2 | 단순화 (아이템 많음) |
| Vault | Rare | 3 | 0 | 0 | 0 | 빈 방 |
| Freezer | Rare | 0 | 0 | 0 | 0 | 빈 방 |
| Her Ladyship's Chamber | Rare | 0 | 0 | 0 | 0 | 빈 방 |
| Laundry Room | Rare | 1 | 0 | 0 | 0 | 빈 방 |
| Furnace | Rare | 0 | 0 | 0 | 0 | 빈 방 (빨간 방) |

### 일자 (14)
| 방 | 희귀도 | 보석 비용 | 걸음 | 열쇠 | 보석 | 근거 |
|---|---|---|---|---|---|---|
| Spare Room | Commonplace | 0 | 0 | 0 | 0 | 원작 (효과 없음) |
| Corridor | Commonplace | 0 | 0 | 0 | 0 | 원작은 "문이 절대 안 잠김" (선택 규칙) |
| Rumpus Room | Standard | 1 | 0 | 0 | 0 | 빈 방 |
| The Kennel | Standard | 0 | 0 | 0 | 0 | 빈 방 |
| Tunnel | Standard | 0 | 0 | 0 | 0 | 빈 방 |
| Workshop | Unusual | 0 | 0 | 0 | 0 | 빈 방 |
| Foyer | Unusual | 2 | 0 | 0 | 0 | 원작은 "복도 방의 문 잠금 해제" (선택 규칙) |
| Ballroom | Unusual | 2 | 0 | 0 | 0 | 빈 방 |
| Secret Passage | Unusual | 1 | 0 | 0 | 0 | 빈 방 (위키 페이지마다 모양이 달라 Room shape 페이지를 따름) |
| Veranda | Unusual | 2 | 0 | 0 | 0 | 빈 방 (원작은 한쪽 문이 늘 바깥, 여기서는 보통 일자) |
| Locker Room | Rare | 1 | 0 | +3 | 0 | 단순화 (집 곳곳에 열쇠를 뿌림) |
| Gallery | Rare | 0 | 0 | 0 | 0 | 빈 방 |
| Drafting Studio | Rare | 2 | 0 | 0 | 0 | 빈 방 |
| Showroom | Rare | 2 | 0 | 0 | 0 | 빈 방 |

### ㄱ자 (27)
방향: 좌 = 들어온 문에서 왼쪽으로 꺾음, 우 = 오른쪽으로 꺾음. 각 희귀도 안에서 좌우를 번갈아 배정했다 (좌 14, 우 13).

| 방 | 방향 | 희귀도 | 보석 비용 | 걸음 | 열쇠 | 보석 | 근거 |
|---|---|---|---|---|---|---|---|
| Bedroom | 좌 | Commonplace | 0 | +2 | 0 | 0 | 원작 (원작은 들어갈 때마다 +2) |
| Parlor | 우 | Commonplace | 0 | 0 | 0 | +1 | 단순화 (퍼즐 보상 보석 2) |
| Billiard Room | 좌 | Commonplace | 0 | 0 | +1 | 0 | 단순화 (퍼즐 보상 열쇠 2) |
| Nook | 우 | Commonplace | 0 | 0 | +1 | 0 | 원작 |
| Pantry | 좌 | Commonplace | 0 | +3 | 0 | 0 | 단순화 (과일 = 걸음) |
| Kitchen | 우 | Commonplace | 1 | 0 | 0 | 0 | 빈 방 |
| Boudoir | 좌 | Standard | 0 | 0 | 0 | +1 | 원작 (금고 보석 1) |
| Office | 우 | Standard | 2 | 0 | 0 | +1 | 원작 (금고 보석 1, 동전은 제거) |
| Laboratory | 좌 | Standard | 1 | 0 | 0 | 0 | 빈 방 |
| Observatory | 우 | Standard | 1 | 0 | 0 | 0 | 빈 방 |
| Patio | 좌 | Standard | 1 | 0 | 0 | +1 | 단순화 (초록 방에 보석 뿌림) |
| Commissary | 우 | Standard | 1 | 0 | 0 | 0 | 빈 방 |
| The Armory | 좌 | Standard | 0 | 0 | 0 | 0 | 빈 방 |
| Library | 우 | Unusual | 0 | 0 | 0 | 0 | 빈 방 |
| Music Room | 좌 | Unusual | 2 | 0 | +2 | 0 | 단순화 (열쇠 1 + 특수 열쇠) |
| Pump Room | 우 | Unusual | 0 | 0 | 0 | 0 | 빈 방 |
| Maid's Chamber | 좌 | Unusual | 0 | 0 | 0 | 0 | 빈 방 (빨간 방) |
| Dovecote | 우 | Unusual | 0 | 0 | 0 | 0 | 빈 방 |
| Classroom | 좌 | Unusual | 1 | 0 | 0 | 0 | 빈 방 |
| Casino | 우 | Unusual | 2 | 0 | 0 | 0 | 빈 방 |
| Conservatory | 좌 | Unusual | 1 | 0 | 0 | 0 | 빈 방 |
| Lost & Found | 우 | Unusual | 0 | 0 | 0 | 0 | 빈 방 (빨간 방) |
| Room 8 | 좌 | Rare | 0 | 0 | 0 | 0 | 빈 방 |
| Trophy Room | 우 | Rare | 5 | 0 | 0 | 0 | 빈 방 |
| Morning Room | 좌 | Rare | 0 | 0 | 0 | 0 | 빈 방 |
| Bookshop | 우 | Rare | 1 | 0 | 0 | 0 | 빈 방 |
| Treasure Trove | 좌 | Rare* | 1 | 0 | 0 | 0 | 빈 방 |

### T자 (20)
| 방 | 희귀도 | 보석 비용 | 걸음 | 열쇠 | 보석 | 근거 |
|---|---|---|---|---|---|---|
| Hallway | Commonplace | 0 | 0 | 0 | 0 | 원작 (효과 없음) |
| Den | Commonplace | 0 | 0 | 0 | +1 | 원작 |
| Drawing Room | Commonplace | 1 | 0 | 0 | +1 | 원작 (금고 보석 1) |
| Chapel | Commonplace | 0 | −2 | 0 | 0 | 빨간 방 (원작은 동전을 뺏음) |
| West Wing Hall | Standard | 0 | 0 | 0 | 0 | 원작 (효과 없음) |
| The Pool | Standard | 1 | 0 | 0 | 0 | 빈 방 |
| Security | Standard | 1 | 0 | 0 | 0 | 빈 방 |
| Dining Room | Standard | 0 | +10 | 0 | 0 | 단순화 (식사 +20, 조건부) |
| Courtyard | Standard | 1 | 0 | 0 | +1 | 단순화 (보석 꽃 2) |
| Gymnasium | Standard | 0 | −2 | 0 | 0 | 원작 (원작은 들어갈 때마다 −2) |
| Darkroom | Standard | 0 | −1 | 0 | 0 | 빨간 방 (원작은 시야 방해) |
| East Wing Hall | Unusual | 0 | 0 | 0 | 0 | 원작 (효과 없음) |
| Conference Room | Unusual | 0 | 0 | 0 | 0 | 빈 방 |
| Boiler Room | Unusual | 1 | 0 | 0 | 0 | 빈 방 |
| Aquarium | Unusual | 1 | 0 | 0 | 0 | 빈 방 |
| The Foundation | Rare | 0 | 0 | 0 | 0 | 빈 방 (원작은 영구 방) |
| Secret Garden | Rare | 0 | 0 | 0 | 0 | 빈 방 |
| Gift Shop | Rare | 0 | 0 | 0 | 0 | 빈 방 |
| Closed Exhibit | Rare | 0 | 0 | 0 | 0 | 빈 방 (빨간 방) |
| Throne Room | Rare* | 5 | 0 | 0 | 0 | 빈 방 |

### 십자 (9)
| 방 | 희귀도 | 보석 비용 | 걸음 | 열쇠 | 보석 | 근거 |
|---|---|---|---|---|---|---|
| Passageway | Commonplace | 2 | 0 | 0 | 0 | 원작 (문이 많은 게 장점) |
| Great Hall | Unusual | 0 | 0 | 0 | 0 | 원작은 문 7개가 잠김 (선택 규칙) |
| Archives | Unusual | 0 | −1 | 0 | 0 | 빨간 방 |
| Cloister | Unusual | 3 | 0 | 0 | +2 | 단순화 (보석 꽃 2) |
| Vestibule | Unusual | 2 | 0 | 0 | 0 | 빈 방 |
| Mechanarium | Unusual | 0 | 0 | 0 | 0 | 빈 방 |
| Rotunda | Rare | 3 | 0 | 0 | 0 | 빈 방 |
| Weight Room | Rare | 0 | −10 | 0 | 0 | 단순화 (원작은 걸음 절반) |
| Chamber of Mirrors | Rare | 0 | 0 | 0 | 0 | 빈 방 (위키 페이지마다 모양이 달라 Room shape 페이지를 따름) |

\* Treasure Trove, Throne Room: 위키에 희귀도가 없어서 Rare로 추정.

## 보상이 있는 방 (보상 많은 순, 27개)
보상 합계 = 걸음·열쇠·보석 개수를 단순히 더한 값. 합계가 같으면 보석 비용이 싼 순. 걸음을 깎는 빨간 방은 제외.

| 순위 | 방 | 모양 | 희귀도 | 보석 비용 | 보상 | 합계 |
|---|---|---|---|---|---|---|
| 1 | Guest Bedroom | 막다른 방 | Commonplace | 0 | 걸음 +10 | 10 |
| 1 | Dining Room | T자 | Standard | 0 | 걸음 +10 | 10 |
| 1 | Master Bedroom | 막다른 방 | Rare | 2 | 걸음 +10 | 10 |
| 4 | Nursery | 막다른 방 | Commonplace | 1 | 걸음 +5 | 5 |
| 5 | Attic | 막다른 방 | Rare | 3 | 열쇠 +2, 보석 +2 | 4 |
| 6 | Pantry | ㄱ자 (좌) | Commonplace | 0 | 걸음 +3 | 3 |
| 6 | Wine Cellar | 막다른 방 | Unusual | 0 | 보석 +3 | 3 |
| 6 | Garage | 막다른 방 | Unusual | 1 | 열쇠 +3 | 3 |
| 6 | Locker Room | 일자 | Rare | 1 | 열쇠 +3 | 3 |
| 10 | Bedroom | ㄱ자 (좌) | Commonplace | 0 | 걸음 +2 | 2 |
| 10 | Storeroom | 막다른 방 | Commonplace | 0 | 열쇠 +1, 보석 +1 | 2 |
| 10 | Walk-In Closet | 막다른 방 | Standard | 1 | 열쇠 +1, 보석 +1 | 2 |
| 10 | Greenhouse | 막다른 방 | Standard | 1 | 보석 +2 | 2 |
| 10 | Servant's Quarters | 막다른 방 | Unusual | 1 | 열쇠 +2 | 2 |
| 10 | Music Room | ㄱ자 (좌) | Unusual | 2 | 열쇠 +2 | 2 |
| 10 | Cloister | 십자 | Unusual | 3 | 보석 +2 | 2 |
| 17 | Closet | 막다른 방 | Commonplace | 0 | 열쇠 +1 | 1 |
| 17 | Billiard Room | ㄱ자 (좌) | Commonplace | 0 | 열쇠 +1 | 1 |
| 17 | Nook | ㄱ자 (우) | Commonplace | 0 | 열쇠 +1 | 1 |
| 17 | Parlor | ㄱ자 (우) | Commonplace | 0 | 보석 +1 | 1 |
| 17 | Den | T자 | Commonplace | 0 | 보석 +1 | 1 |
| 17 | Boudoir | ㄱ자 (좌) | Standard | 0 | 보석 +1 | 1 |
| 17 | Study | 막다른 방 | Unusual | 0 | 보석 +1 | 1 |
| 17 | Drawing Room | T자 | Commonplace | 1 | 보석 +1 | 1 |
| 17 | Patio | ㄱ자 (좌) | Standard | 1 | 보석 +1 | 1 |
| 17 | Courtyard | T자 | Standard | 1 | 보석 +1 | 1 |
| 17 | Office | ㄱ자 (우) | Standard | 2 | 보석 +1 | 1 |

## 목표 방 (Antechamber)
- 맨 위 가운데(9행 C열)에 고정. 문은 남·서·동 3개.
- 바깥에서 열 수 없고, 새 방의 문을 Antechamber의 문에 이어 붙여야 길이 생긴다.
- **들어갈 때 열쇠 3개를 쓴다.** (확정)

## 목록에 없는 방
| 방 | 이유 |
|---|---|
| Entrance Hall, Antechamber | 시작 방·목표 방으로 따로 고정 |
| Room 46 | 모양 정보가 없음 (원작에서도 특수한 목표 방) |
| 바깥 방 (Toolshed, Shelter 등 8개) | 집 격자 밖에 있는 방 |
| 업그레이드 변형 (Parlor 업그레이드, Goldfish Aquarium 등) | 기본 방과 같은 방의 변형 |

## 관찰
- 99개 중 **약 60개가 빈 방**이다. 덱 대부분이 "모양 + 보석 비용"만 다른 연결용 방이 되므로, 자원을 주는 방(약 30개)이 상대적으로 귀해진다. 이게 오히려 "보상 방을 노릴지, 길을 넓힐지"라는 선택을 선명하게 만든다.
- AI 입장에서 실제로 다른 방은 (모양, ㄱ자 방향, 보석 비용, 보상) 조합이라 약 30종이다. 이름은 분위기용.
- 빈 방이면서 보석이 비싼 방(Trophy Room 5, Throne Room 5, Vault 3, Rotunda 3)은 고를 이유가 거의 없다. 첫 칸 무료 규칙 때문에 가끔 공짜로만 쓰인다.
- 빨간 방 중 Lavatory, Maid's Chamber, Lost & Found, Closed Exhibit, Furnace는 빈 방으로, Chapel, Gymnasium, Darkroom, Archives, Weight Room은 걸음 손해 방으로 남겼다.

## 정해야 할 것
1. 자원을 놓을 때 한 번 받을지, 원작처럼 일부(Bedroom, Gymnasium)는 들어갈 때마다 받을지.
2. ~~덱에서 방마다 몇 장씩 넣을지~~ → **확정: 모든 방은 덱에 딱 1장.** 놓은 방은 그 판 동안 덱에서 빠진다 (덱 99장).
3. 희귀도별 뽑힐 확률 (위키에 수치 없음. 예: 흔함 50 / 보통 30 / 드묾 15 / 희귀 5, 위쪽 행일수록 희귀 증가).
4. 선택 규칙(Corridor 문 안 잠김, Foyer, Great Hall 잠김)을 넣을지.
5. 비싼 빈 방(Trophy Room, Throne Room, Vault, Rotunda)과 비싸기만 한 방(Passageway, Foyer)의 비용을 낮출지.
6. 빈 빨간 방(Lavatory 등 5개)도 다른 빨간 방처럼 걸음 손해를 줄지.
