

def evaluation4class(prediction, y):  # 4 dim
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0
    TP2, FP2, FN2, TN2 = 0, 0, 0, 0
    TP3, FP3, FN3, TN3 = 0, 0, 0, 0
    TP4, FP4, FN4, TN4 = 0, 0, 0, 0
    # e, RMSE, RMSE1, RMSE2, RMSE3, RMSE4 = 0.000001, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(len(y)):
        Act, Pre = y[i], prediction[i]

        ## for class 1
        if Act == 0 and Pre == 0: TP1 += 1
        if Act == 0 and Pre != 0: FN1 += 1
        if Act != 0 and Pre == 0: FP1 += 1
        if Act != 0 and Pre != 0: TN1 += 1
        ## for class 2
        if Act == 1 and Pre == 1: TP2 += 1
        if Act == 1 and Pre != 1: FN2 += 1
        if Act != 1 and Pre == 1: FP2 += 1
        if Act != 1 and Pre != 1: TN2 += 1
        ## for class 3
        if Act == 2 and Pre == 2: TP3 += 1
        if Act == 2 and Pre != 2: FN3 += 1
        if Act != 2 and Pre == 2: FP3 += 1
        if Act != 2 and Pre != 2: TN3 += 1
        ## for class 4
        if Act == 3 and Pre == 3: TP4 += 1
        if Act == 3 and Pre != 3: FN4 += 1
        if Act != 3 and Pre == 3: FP4 += 1
        if Act != 3 and Pre != 3: TN4 += 1

    ## print result
    Acc_all = round(float(TP1 + TP2 + TP3 + TP4) / float(len(y) ), 4)
    Acc1 = round(float(TP1 + TN1) / float(TP1 + TN1 + FN1 + FP1), 4)
    if (TP1 + FP1)==0:
        Prec1 =0
    else:
        Prec1 = round(float(TP1) / float(TP1 + FP1), 4)
    if (TP1 + FN1 )==0:
        Recll1 =0
    else:
        Recll1 = round(float(TP1) / float(TP1 + FN1 ), 4)
    if (Prec1 + Recll1 )==0:
        F1 =0
    else:
        F1 = round(2 * Prec1 * Recll1 / (Prec1 + Recll1 ), 4)

    Acc2 = round(float(TP2 + TN2) / float(TP2 + TN2 + FN2 + FP2), 4)
    if (TP2 + FP2)==0:
        Prec2 =0
    else:
        Prec2 = round(float(TP2) / float(TP2 + FP2), 4)
    if (TP2 + FN2 )==0:
        Recll2 =0
    else:
        Recll2 = round(float(TP2) / float(TP2 + FN2 ), 4)
    if (Prec2 + Recll2 )==0:
        F2 =0
    else:
        F2 = round(2 * Prec2 * Recll2 / (Prec2 + Recll2 ), 4)

    Acc3 = round(float(TP3 + TN3) / float(TP3 + TN3 + FN3 + FP3), 4)
    if (TP3 + FP3)==0:
        Prec3 =0
    else:
        Prec3 = round(float(TP3) / float(TP3 + FP3), 4)
    if (TP3 + FN3 )==0:
        Recll3 =0
    else:
        Recll3 = round(float(TP3) / float(TP3 + FN3), 4)
    if (Prec3 + Recll3 )==0:
        F3 =0
    else:
        F3 = round(2 * Prec3 * Recll3 / (Prec3 + Recll3), 4)

    Acc4 = round(float(TP4 + TN4) / float(TP4 + TN4 + FN4 + FP4), 4)
    if (TP4 + FP4)==0:
        Prec4 =0
    else:
        Prec4 = round(float(TP4) / float(TP4 + FP4), 4)
    if (TP4 + FN4) == 0:
        Recll4 = 0
    else:
        Recll4 = round(float(TP4) / float(TP4 + FN4), 4)
    if (Prec4 + Recll4 )==0:
        F4 =0
    else:
        F4 = round(2 * Prec4 * Recll4 / (Prec4 + Recll4), 4)

    return  Acc_all,Acc1, Prec1, Recll1, F1,Acc2, Prec2, Recll2, F2,Acc3, Prec3, Recll3, F3,Acc4, Prec4, Recll4, F4

def evaluationclass(prediction, y):  # 2 dim
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0
    TP2, FP2, FN2, TN2 = 0, 0, 0, 0
    for i in range(len(y)):
        Act, Pre = y[i], prediction[i]

        ## for class 1
        if Act == 0 and Pre == 0: TP1 += 1
        if Act == 0 and Pre != 0: FN1 += 1
        if Act != 0 and Pre == 0: FP1 += 1
        if Act != 0 and Pre != 0: TN1 += 1
        ## for class 2
        if Act == 1 and Pre == 1: TP2 += 1
        if Act == 1 and Pre != 1: FN2 += 1
        if Act != 1 and Pre == 1: FP2 += 1
        if Act != 1 and Pre != 1: TN2 += 1

    ## print result
    Acc_all = round(float(TP1 + TP2) / float(len(y) ), 4)
    Acc1 = round(float(TP1 + TN1) / float(TP1 + TN1 + FN1 + FP1), 4)
    if (TP1 + FP1)==0:
        Prec1 =0
    else:
        Prec1 = round(float(TP1) / float(TP1 + FP1), 4)
    if (TP1 + FN1 )==0:
        Recll1 =0
    else:
        Recll1 = round(float(TP1) / float(TP1 + FN1 ), 4)
    if (Prec1 + Recll1 )==0:
        F1 =0
    else:
        F1 = round(2 * Prec1 * Recll1 / (Prec1 + Recll1 ), 4)

    Acc2 = round(float(TP2 + TN2) / float(TP2 + TN2 + FN2 + FP2), 4)
    if (TP2 + FP2)==0:
        Prec2 =0
    else:
        Prec2 = round(float(TP2) / float(TP2 + FP2), 4)
    if (TP2 + FN2 )==0:
        Recll2 =0
    else:
        Recll2 = round(float(TP2) / float(TP2 + FN2 ), 4)
    if (Prec2 + Recll2 )==0:
        F2 =0
    else:
        F2 = round(2 * Prec2 * Recll2 / (Prec2 + Recll2 ), 4)

    return  Acc_all,Acc1, Prec1, Recll1, F1,Acc2, Prec2, Recll2, F2