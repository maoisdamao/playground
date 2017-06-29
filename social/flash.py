import time
import sys
import itchat
from itchat.content import TEXT


@itchat.msg_register(TEXT)
def snap(msg, toUserName, delay=5):
    res = itchat.send(msg=msg, toUserName=toUserName)
    msgID = res['MsgID']
    time.sleep(delay)
    revoke_res = itchat.revoke(msgID, toUserName)
    print(revoke_res)


if __name__ == '__main__':
    itchat.auto_login(hotReload=True, enableCmdQR=2)
    userType = input("send to user or chatroom?[user/chatroom]")
    if userType == 'user':
        remark_name = input("send to who?")
        to_user = itchat.search_friends(remarkName=remark_name)
        username = to_user[0].UserName
    elif userType == 'chatroom':
        room_name = input("send to which chatroom?")
        chatrooms = itchat.search_chatrooms(name=room_name)
        username = chatrooms[0].UserName
    else:
        sys.exit()
    msg = input("input your msg:")
    snap(msg, username)
