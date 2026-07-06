# [2026_DS_RL] HW2 - Raw Extracted Text

## Page 1

[2026 Samsung DS - RL] 과제 2 
 
제출 마감일: 2026년 7월 17일 (금) 23:59 
제출 방법: 구글 클래스룸 과제란 제출 
 
🗂️ 환경 정보 
2026Spring_RL_Lab/HW2/configs/ 경로에 세 개의 맵 (hw_map1.yaml, hw_map2.yaml, 
hw_map3.yaml) 이 제공됩니다. 제공된 맵을 기반으로 Policy Gradient 을 수정·적용하여 높은 
성공률을 달성하는 것이 목표입니다. 
 
<HW map, 왼쪽부터 난이도 하, 중, 상> 
 
1. REINFORCE 알고리즘 (총점의 70%) 
실습 시간에 사용한 REINFORCE 코드에서 다음 조건을 만족하도록 파라미터를 조정하거나 알고리즘을 
수정합니다. 제공된 REINFORCE 코드는 가장 기본적인 형태의 알고리즘이며, 기본 알고리즘으로는 
제공된 맵을 해결하기 어려울 수 있기 때문에 수업시간에 배우신 기법들을 다양하게 접목해 보시는 것을 
권장드립니다.: 
●​ 최대 에피소드 수 (max_episodes) : 200000 
○​ 최대 상한선을 제한한 것이며, 이보다 적은 수에 학습되어도 좋습니다. 
●​ 에피소드당 최대 스텝 수 (max_steps) :  
○​ hw_map1 : 150 
○​ hw_map2 : 200 
○​ hw_map3 : 250 
❖​ 코드 실행 방법 


---

## Page 2

-​
학습: 
python train_r.py --map [yaml 파일명] --episodes [에피소드 수] --max-steps [스텝 수] 
-​
테스트 (렌더링): 
​
python test_r.py --map [yaml 파일명] --model [checkpoint .pth 파일명] 
 
❖​ 변경 및 적용 가능 요소 
➢​ 가능 요소 
■​
Hyper parameter (epsilon, learning rate, discount factor 등) 
■​
Network size 및 layer 수 (변경은 가능하지만 너무 크게는 권장드리지는 않습니다.) 
■​
Policy Gradient 기반의 알고리즘이라면 구현하여 적용 가능합니다. (ex. Actor Critic, 
N-step, TD(lambda) 등) 
➢​ 불가능 요소 
■​
외부 RL 라이브러리를 사용하는 것은 제한합니다. (ex. stable baseline) 
■​
환경에 대한 요소를 수정하는 것은 제한합니다. (step size, reward 등)​
(만약 환경에 대한 버그가 발생할 경우, 조교에게 문의 바랍니다.) 
 
2. 보고서 (총점의 30%) 
A4 기준 1-2페이지 이내의 보고서를 작성하며, 아래 내용을 포함합니다: 
●​ REINFORCE 알고리즘에서 수정한 내용과 그 이유 
●​ 자체 실험 결과 분석 및 설명 (ex. Tensorboard 시각화 그래프 등) 
●​ 실험 방법에 대한 부가 설명 (시도 횟수, 성공률, 알고리즘의 한계점 등) 
 
[채점 기준] 
조교 컴퓨터에서 동일한 조건으로 각 맵에 알고리즘을 10회 실행하여 성능을 평가합니다. 맵은 난이도 
별로 배점이 상이하며 각 맵의 점수는 [하 : 15점, 중 : 25점, 상 : 30점] 입니다. 
●​ 5회 이상 성공: 기본 통과 (배점의 50%) 
●​ 8회 이상 성공: 우수 통과 (배점의 100%) 
●​ 5회 미만 성공: 수정한 알고리즘의 내용을 바탕으로 정성적으로 평가하여 점수 부여 
 
[제출물] 
●​ 수정된 알고리즘 코드 및 checkpoint 파일 (.pth) 
●​ 보고서 (PDF 형식) 
