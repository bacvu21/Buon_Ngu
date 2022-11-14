from tham_so import *
from scipy.spatial import distance
from imutils import face_utils as face
import imutils
import time
import dlib
import cv2
from datetime import datetime
from playsound import playsound


#Hàm vẽ các khung hình trên khuôn mặt 
def get_max_area_rect(rects):
    # Kiểm tra xem có khuôn mặt hay không !
    if len(rects)==0: return
    areas=[]
    for rect in rects:
        areas.append(rect.area())
    return rects[areas.index(max(areas))]

#hàm tính toán tỉ lệ khung hình mắt 
def get_eye_aspect_ratio(eye): 
    # các điểm đánh dấu  mắt với tọa độ (x,y) trên landmarks
    vertical_1 = distance.euclidean(eye[1], eye[5]) # các điểm nằm dọc theo mắt 
    vertical_2 = distance.euclidean(eye[2], eye[4]) # các điểm nằm dọc theo mắt 
    horizontal = distance.euclidean(eye[0], eye[3]) # các điểm nằm ngnag theo mắt 
    #returns EAR
    return (vertical_1+vertical_2)/(horizontal*2)


#Hàm tính toán tỉ lệ khung hình miệng
def get_mouth_aspect_ratio(mouth):
    # các điểm đánh dấu miệng trên file land mark
    horizontal=distance.euclidean(mouth[0],mouth[4]) # các điểm nằm ngang miệng 
    vertical=0
    for coord in range(1,4):
        vertical+=distance.euclidean(mouth[coord],mouth[8-coord])
    #return MAR
    return vertical/(horizontal*3)


# Hàm xử lí khuôn mặt 
def facial_processing():
    distracton_initialized = False # Biến boolean để tính so sánh mức độ xao nhãng 
    eye_initialized      = False
    mouth_initialized    = False
    normal_initialized   = False
	
    #Lấy dữ liệu các điểm đánh dấu trên khuôn mặt qua file 68 face landmarks sử dụng thuật toán trong dlib pack
    #dlib là một công cụ trong C++ chứa các thuật toán nhận diện khuôn mặt để giải quyết các vấn đề về thế giới thực .
    detector    = dlib.get_frontal_face_detector()
    predictor   = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   #sử dụng để chuyển thành các điểm trên khuôn mặt 



    # Lấy các chỉ số của các mốc trên khuôn mặt cho mắt trái và mắt phải, tương ứng
    ls,le = face.FACIAL_LANDMARKS_IDXS["left_eye"]
    rs,re = face.FACIAL_LANDMARKS_IDXS["right_eye"]

    #sử dung webcam của máy tính 
    cap=cv2.VideoCapture(0)
    
    #Đếm số khung hình 
    fps_counter=0
    fps_to_display='Dang nhan dien.....'
    fps_timer=time.time()
    # vòng lặp để chuyển đổi ảnh tại các khung hình tại web cam 
    
    while True:
        _ , frame=cap.read()
        fps_counter+=1
        
	#Lật frame theo trục y để cho ảnh giống trong gương :D
        frame = cv2.flip(frame, 1)
        if time.time()-fps_timer>=1.0:
            fps_to_display=fps_counter
            fps_timer=time.time()
            fps_counter=0
            
	#Hiển thị fps lên màn hình 
        cv2.putText(frame, "FPS :"+str(fps_to_display), (frame.shape[1]-100, frame.shape[0]-10),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


	#Chuyển đổi ảnh sang khung hình xám và sử dụng thang đo màu Blue green red để sử dụng cho thuật toán CNN
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	
	# phát hiện khuôn mặt trong khung hình xám 
        rects = detector(gray, 0)
        
        
	#Vẽ các bouding box quanh khuôn mặt khi nhận ra 
        rect=get_max_area_rect(rects)

# khi khung hình k được phát hiện 
        if rect!=None:
            #đo thời gian mắt người dùng khi rời khỏi đường 
            if distracton_initialized==True:    
                interval=time.time()-distracton_start_time # Khoảng thời gian bằng thời gian hàm thời gian - thời gian rời khỏi màn hình 
                interval=str(round(interval,3))  # khoảng thời gian sẽ bằng một giá trị thập phân được tính bên trên lấy giá trị thập phân dưới 3 con số 
                
                
		#lay gia trị hiện tại dựa vào thư viện thời gian trong máy tính 
                dateTime= datetime.now()
                distracton_initialized=False
                info="Ngay: " + str(dateTime) + ", Khoangthoigian: " + interval + ", Type: Ban dang mat tap trung khi lai xe !"
                info=info+ "\n"
                if time.time()- distracton_start_time> DISTRACTION_INTERVAL:
		   #Lưu giá trị vào file output.txt
     
                    with open(r'output.txt', "a+") as file_object:
                        file_object.write(info)
			
	    # xác định các vùng trên khuôn mặt , sau đó chuyển đổi tọa độ khuôn mặt (x, y) thành mảng NumPy
            shape = predictor(gray, rect)    #dựa vào file 68 landmark và hình ảnh webcam xác định khuôn mặt 
            shape = face.shape_to_np(shape)
		
	    # trích xuất tọa độ mắt trái và phải, sau đó sử dụng
	    # tọa độ để tính tỷ lệ khung hình mắt cho cả hai mắt
            leftEye = shape[ls:le]
            rightEye = shape[rs:re]
	    #Lấy tọa độ khung hình cho mỗi mắt 
            leftEAR = get_eye_aspect_ratio(leftEye)  #tọa độ tại mắt trái sử dụng hàm get_eye 
            rightEAR = get_eye_aspect_ratio(rightEye) #tọa độ tại mắt phải 
        # lấy tọa độ miệng 
            inner_lips=shape[60:68]   
            mar=get_mouth_aspect_ratio(inner_lips)

	
	    # trung bình tỷ lệ khung hình mắt với nhau cho cả hai mắt
            eye_aspect_ratio = (leftEAR + rightEAR) / 2.0

            
        # Tính toán phần nhận ra ở cả hai mắt trái và phải , sau đó
	    #  vẽ các hình giới hạn quanh mắt
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)
            lipHull = cv2.convexHull(inner_lips)
            cv2.drawContours(frame, [lipHull], -1, (255, 255, 255), 1)

	    #Hien thị các giá trị tỉ lệ khung hình
            cv2.putText(frame, "khung mat: {:.2f} khung mieng:{:.2f}".format(eye_aspect_ratio,mar), (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            
	    #kiểm tra mắt có nhắm vào hay không 
            if eye_aspect_ratio < EYE_DROWSINESS_THRESHOLD:

                if not eye_initialized: 
                    eye_start_time= time.time() #thời gian mắt bắt đầu nhắm
                    eye_initialized=True
                    
                    
		#Kiểm tra xem mắt có buồn ngủ trên các khung hình không
                if time.time()-eye_start_time >= EYE_DROWSINESS_INTERVAL:  #nếu giá trị thời gian nhắm >= khoảng cách mắt buồn ngủ 
                    cv2.putText(frame, "Ban dang buon ngu!!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    playsound('Ngugat.mp3')

                    dateTimeOBJ=datetime.now()
                    eye_info="Ngay: " + str(dateTimeOBJ) + " Khoangtg: " + str(time.time()-eye_start_time ) + "Dang buon ngu!!"
                    print(eye_info)

            else:
                #đo khoảng thời gian mà mắt người dùng buồn ngủ
                if eye_initialized==True:
                    interval_eye=time.time()-eye_start_time
                    interval_eye=str(round(interval_eye,3))
                    dateTime_eye= datetime.now()
                    eye_initialized=False
                    info_eye="ngay: " + str(dateTime_eye) + ",Khoangtg: " + interval_eye + ",Giatri:Dang buong ngu!!"
                    info_eye=info_eye+ "\n"
		    # chỉ lưu trữ thông tin nếu mắt người dùng nhắm mắt trong một khoảng thời gian, đủ
                    if time.time()-eye_start_time >= EYE_DROWSINESS_INTERVAL:
                        
                        
			#lưu giá trị ra file txt 
                        with open(r'output.txt', "a+") as file_object:
                            file_object.write(info_eye)



	    #Kiem ta nguoi dùng đang ngáp ngủ 
            if mar > MOUTH_DROWSINESS_THRESHOLD: #kiểm tra bằng các điểm đnáh dấu trên miệng

                if not mouth_initialized:
                    mouth_start_time= time.time()
                    mouth_initialized=True
                    
		#kiểm tra người dùng ngáp trung khoảng thời gian từng khung hình 
                if time.time()-mouth_start_time >= MOUTH_DROWSINESS_INTERVAL:
                    cv2.putText(frame, "BAN DANG NGAP NGU!!", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    playsound('Ngap.mp3')
                   
                    dateTimeOBJ2=datetime.now()
                    mouth_info="Ngay: " + str(dateTimeOBJ2) + "Khoangtg: " + str(time.time()-mouth_start_time ) + " Ngap Ngu" + " Ti Le " + str(mar)
                    print(mouth_info)

            else:
                
                #Đo khoảng cách miệng 
                if mouth_initialized==True:
                    interval_mouth=time.time()-mouth_start_time
                    interval_mouth=str(round(interval_mouth,3))
                    dateTime_mouth= datetime.now()
                    mouth_initialized=False
                    info_mouth="Ngay: " + str(dateTime_mouth) + ",Khoangtg: " + interval_mouth + ", Giatri :Dang ngap ngu!!"
                    info_mouth=info_mouth+ "\n"
		   
                    if time.time()-mouth_start_time >= MOUTH_DROWSINESS_INTERVAL:
			#lưu ra file txt
                        with open(r'output.txt', "a+") as file_object:
                            file_object.write(info_mouth)


            #Kiểm tra xem người dùng có đang tập trung lái xe
            if (eye_initialized==False) & (mouth_initialized==False) & (distracton_initialized==False):

                if not normal_initialized:
                    normal_start_time= time.time()
                    normal_initialized=True

		#kiểm tra người dùng đang tập trung lái xe theo thời gian và từng khung hình , nếu có thì in ra bình thường
                if time.time()-normal_start_time >= NORMAL_INTERVAL:
                        cv2.putText(frame, "Trang thai binh thuong!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        print('Binh Thuong')
            else:
                if normal_initialized==True:
                    interval_normal=time.time()-normal_start_time
                    interval_normal=str(round(interval_normal,3))
                    dateTime_normal= datetime.now()
                    normal_initialized=False
                    info_normal="Ngay: " + str(dateTime_normal) + ",Khoangtg: " + interval_normal+ ", Giatri: Binh thuong"
                    info_normal=info_normal+ "\n"
		  
                    if time.time()-normal_start_time >= NORMAL_INTERVAL:
                        with open(r'output.txt', "a+") as file_object:
                            file_object.write(info_normal)


	#Nếu người dùng k tập trung trên đường thì các giá trị mắt và miệng không thể tính toán 
        else:
            # nếu người dùng quay đi và mắt họ không còn trên đường nữa
            #nhưng lúc đầu có vẻ như là buồn ngủ , câu lệnh if sẽ kiểm tra và phát hiện buồn ngủ và cảnh báo cho người lái xe tập trung chú ý lái xe 
            if eye_initialized==True:
                    interval_eye=time.time()-eye_start_time
                    interval_eye=str(round(interval_eye,3))
                    dateTime_eye= datetime.now()
                    eye_initialized=False
                    info_eye="Ngay: " + str(dateTime_eye) + ", Khoangtg: " + interval_eye + ", Giatri : Buon ngu"
                    info_eye=info_eye+ "\n"
                    if time.time()-eye_start_time >= EYE_DROWSINESS_INTERVAL:
                        with open(r'output.txt', "a+") as file_object:
                            file_object.write(info_eye)

            if not distracton_initialized:
                distracton_start_time=time.time()
                distracton_initialized=True
              
              
            if time.time()- distracton_start_time> DISTRACTION_INTERVAL:
		#Hien thi cho nguoi lai xe hay chu y lai xe 
                cv2.putText(frame, "HAY CHU Y LAI XE !", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                playsound('Khongchuy.mp3')
                
                #dateTimeOBJ3=datetime.now()
                #DIST_info="date: " + str(dateTimeOBJ3) + " Interval: " + str(time.time()-distracton_start_time) + " EYES NOT ON ROAD"
                #print(DIST_info)

	#Hiện thị 
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(5)&0xFF
	
	# Ấn q để thoát chương trình khung hình
        if key == ord("q"):
            break
	
    # close all windows and release the capture
    cv2.destroyAllWindows()
    cap.release()


if __name__=='__main__':
	facial_processing()