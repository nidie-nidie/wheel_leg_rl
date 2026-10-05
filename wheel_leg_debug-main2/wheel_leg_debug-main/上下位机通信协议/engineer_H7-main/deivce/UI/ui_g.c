//
// Created by RM UI Designer
// Dynamic Edition
//

#include "string.h"
#include "ui_interface.h"
#include "ui_g.h"

#define TOTAL_FIGURE 11
#define TOTAL_STRING 5

ui_interface_figure_t ui_g_now_figures[TOTAL_FIGURE];
ui_interface_string_t ui_g_now_strings[TOTAL_STRING];
uint8_t ui_g_dirty_figure[TOTAL_FIGURE];
uint8_t ui_g_dirty_string[TOTAL_STRING];
#ifndef MANUAL_DIRTY
ui_interface_figure_t ui_g_last_figures[TOTAL_FIGURE];
ui_interface_string_t ui_g_last_strings[TOTAL_STRING];
#endif

void ui_init_g() {
    ui_g_Ungroup_HEIGHT_LINE->figure_tpye = 0;
    ui_g_Ungroup_HEIGHT_LINE->layer = 0;
    ui_g_Ungroup_HEIGHT_LINE->start_x = 1343;
    ui_g_Ungroup_HEIGHT_LINE->start_y = 430;
    ui_g_Ungroup_HEIGHT_LINE->end_x = 1345;
    ui_g_Ungroup_HEIGHT_LINE->end_y = 844;
    ui_g_Ungroup_HEIGHT_LINE->color = 2;
    ui_g_Ungroup_HEIGHT_LINE->width = 3;

    ui_g_Ungroup_ARROW1->figure_tpye = 0;
    ui_g_Ungroup_ARROW1->layer = 0;
    ui_g_Ungroup_ARROW1->start_x = 1313;
    ui_g_Ungroup_ARROW1->start_y = 810;
    ui_g_Ungroup_ARROW1->end_x = 1347;
    ui_g_Ungroup_ARROW1->end_y = 844;
    ui_g_Ungroup_ARROW1->color = 2;
    ui_g_Ungroup_ARROW1->width = 3;

    ui_g_Ungroup_ARROW2->figure_tpye = 0;
    ui_g_Ungroup_ARROW2->layer = 0;
    ui_g_Ungroup_ARROW2->start_x = 1344;
    ui_g_Ungroup_ARROW2->start_y = 845;
    ui_g_Ungroup_ARROW2->end_x = 1378;
    ui_g_Ungroup_ARROW2->end_y = 813;
    ui_g_Ungroup_ARROW2->color = 2;
    ui_g_Ungroup_ARROW2->width = 3;

    ui_g_Ungroup_heigth_index->figure_tpye = 2;
    ui_g_Ungroup_heigth_index->layer = 0;
    ui_g_Ungroup_heigth_index->r = 10;
    ui_g_Ungroup_heigth_index->start_x = 1340;
    ui_g_Ungroup_heigth_index->start_y = 430;
    ui_g_Ungroup_heigth_index->color = 4;
    ui_g_Ungroup_heigth_index->width = 12;

    ui_g_Ungroup_REAL_HEIGHT->figure_tpye = 6;
    ui_g_Ungroup_REAL_HEIGHT->layer = 0;
    ui_g_Ungroup_REAL_HEIGHT->font_size = 20;
    ui_g_Ungroup_REAL_HEIGHT->start_x = 1018;
    ui_g_Ungroup_REAL_HEIGHT->start_y = 810;
    ui_g_Ungroup_REAL_HEIGHT->color = 0;
    ui_g_Ungroup_REAL_HEIGHT->number = 0;
    ui_g_Ungroup_REAL_HEIGHT->width = 2;

    ui_g_Ungroup_SOURCE_LINE1->figure_tpye = 0;
    ui_g_Ungroup_SOURCE_LINE1->layer = 0;
    ui_g_Ungroup_SOURCE_LINE1->start_x = 733;
    ui_g_Ungroup_SOURCE_LINE1->start_y = 125;
    ui_g_Ungroup_SOURCE_LINE1->end_x = 789;
    ui_g_Ungroup_SOURCE_LINE1->end_y = 451;
    ui_g_Ungroup_SOURCE_LINE1->color = 2;
    ui_g_Ungroup_SOURCE_LINE1->width = 3;

    ui_g_Ungroup_SOURCE_LINE2->figure_tpye = 0;
    ui_g_Ungroup_SOURCE_LINE2->layer = 0;
    ui_g_Ungroup_SOURCE_LINE2->start_x = 1181;
    ui_g_Ungroup_SOURCE_LINE2->start_y = 112;
    ui_g_Ungroup_SOURCE_LINE2->end_x = 1108;
    ui_g_Ungroup_SOURCE_LINE2->end_y = 443;
    ui_g_Ungroup_SOURCE_LINE2->color = 2;
    ui_g_Ungroup_SOURCE_LINE2->width = 3;

    ui_g_Ungroup_SILVER1_LOCATION->figure_tpye = 0;
    ui_g_Ungroup_SILVER1_LOCATION->layer = 0;
    ui_g_Ungroup_SILVER1_LOCATION->start_x = 442;
    ui_g_Ungroup_SILVER1_LOCATION->start_y = 419;
    ui_g_Ungroup_SILVER1_LOCATION->end_x = 559;
    ui_g_Ungroup_SILVER1_LOCATION->end_y = 419;
    ui_g_Ungroup_SILVER1_LOCATION->color = 1;
    ui_g_Ungroup_SILVER1_LOCATION->width = 4;

    ui_g_Ungroup_SILVER2_LOCATION->figure_tpye = 0;
    ui_g_Ungroup_SILVER2_LOCATION->layer = 0;
    ui_g_Ungroup_SILVER2_LOCATION->start_x = 889;
    ui_g_Ungroup_SILVER2_LOCATION->start_y = 421;
    ui_g_Ungroup_SILVER2_LOCATION->end_x = 1006;
    ui_g_Ungroup_SILVER2_LOCATION->end_y = 421;
    ui_g_Ungroup_SILVER2_LOCATION->color = 1;
    ui_g_Ungroup_SILVER2_LOCATION->width = 4;

    ui_g_Ungroup_NewRound->figure_tpye = 2;
    ui_g_Ungroup_NewRound->layer = 0;
    ui_g_Ungroup_NewRound->r = 22;
    ui_g_Ungroup_NewRound->start_x = 959;
    ui_g_Ungroup_NewRound->start_y = 536;
    ui_g_Ungroup_NewRound->color = 0;
    ui_g_Ungroup_NewRound->width = 3;

    ui_g_Ungroup_SOURCE_HEIGHT->figure_tpye = 0;
    ui_g_Ungroup_SOURCE_HEIGHT->layer = 0;
    ui_g_Ungroup_SOURCE_HEIGHT->start_x = 820;
    ui_g_Ungroup_SOURCE_HEIGHT->start_y = 746;
    ui_g_Ungroup_SOURCE_HEIGHT->end_x = 1097;
    ui_g_Ungroup_SOURCE_HEIGHT->end_y = 746;
    ui_g_Ungroup_SOURCE_HEIGHT->color = 0;
    ui_g_Ungroup_SOURCE_HEIGHT->width = 4;

    ui_g_Ungroup_SUCKER->figure_tpye = 7;
    ui_g_Ungroup_SUCKER->layer = 0;
    ui_g_Ungroup_SUCKER->font_size = 20;
    ui_g_Ungroup_SUCKER->start_x = 837;
    ui_g_Ungroup_SUCKER->start_y = 902;
    ui_g_Ungroup_SUCKER->color = 3;
    ui_g_Ungroup_SUCKER->str_length = 6;
    ui_g_Ungroup_SUCKER->width = 2;
    strcpy(ui_g_Ungroup_SUCKER->string, "SUCKER");

    ui_g_Ungroup_SUCKER_STATE->figure_tpye = 7;
    ui_g_Ungroup_SUCKER_STATE->layer = 0;
    ui_g_Ungroup_SUCKER_STATE->font_size = 20;
    ui_g_Ungroup_SUCKER_STATE->start_x = 1018;
    ui_g_Ungroup_SUCKER_STATE->start_y = 550;
    ui_g_Ungroup_SUCKER_STATE->color = 3;
    ui_g_Ungroup_SUCKER_STATE->str_length = 3;
    ui_g_Ungroup_SUCKER_STATE->width = 2;
    strcpy(ui_g_Ungroup_SUCKER_STATE->string, "OFF");

    ui_g_Ungroup_ROBOT->figure_tpye = 7;
    ui_g_Ungroup_ROBOT->layer = 0;
    ui_g_Ungroup_ROBOT->font_size = 20;
    ui_g_Ungroup_ROBOT->start_x = 743;
    ui_g_Ungroup_ROBOT->start_y = 855;
    ui_g_Ungroup_ROBOT->color = 3;
    ui_g_Ungroup_ROBOT->str_length = 11;
    ui_g_Ungroup_ROBOT->width = 2;
    strcpy(ui_g_Ungroup_ROBOT->string, "ROBOT_STATE");

    ui_g_Ungroup_ROBOT_STATE->figure_tpye = 7;
    ui_g_Ungroup_ROBOT_STATE->layer = 0;
    ui_g_Ungroup_ROBOT_STATE->font_size = 20;
    ui_g_Ungroup_ROBOT_STATE->start_x = 1016;
    ui_g_Ungroup_ROBOT_STATE->start_y = 857;
    ui_g_Ungroup_ROBOT_STATE->color = 3;
    ui_g_Ungroup_ROBOT_STATE->str_length = 6;
    ui_g_Ungroup_ROBOT_STATE->width = 2;
    strcpy(ui_g_Ungroup_ROBOT_STATE->string, "NORMAL");

    ui_g_Ungroup_HEIGHT_NAME->figure_tpye = 7;
    ui_g_Ungroup_HEIGHT_NAME->layer = 0;
    ui_g_Ungroup_HEIGHT_NAME->font_size = 20;
    ui_g_Ungroup_HEIGHT_NAME->start_x = 842;
    ui_g_Ungroup_HEIGHT_NAME->start_y = 812;
    ui_g_Ungroup_HEIGHT_NAME->color = 0;
    ui_g_Ungroup_HEIGHT_NAME->str_length = 6;
    ui_g_Ungroup_HEIGHT_NAME->width = 2;
    strcpy(ui_g_Ungroup_HEIGHT_NAME->string, "HEIGHT");


    uint32_t idx = 0;
    for (int i = 0; i < TOTAL_FIGURE; i++) {
        ui_g_now_figures[i].figure_name[2] = idx & 0xFF;
        ui_g_now_figures[i].figure_name[1] = (idx >> 8) & 0xFF;
        ui_g_now_figures[i].figure_name[0] = (idx >> 16) & 0xFF;
        ui_g_now_figures[i].operate_tpyel = 1;
#ifndef MANUAL_DIRTY
        ui_g_last_figures[i] = ui_g_now_figures[i];
#endif
        ui_g_dirty_figure[i] = 1;
        idx++;
    }
    for (int i = 0; i < TOTAL_STRING; i++) {
        ui_g_now_strings[i].figure_name[2] = idx & 0xFF;
        ui_g_now_strings[i].figure_name[1] = (idx >> 8) & 0xFF;
        ui_g_now_strings[i].figure_name[0] = (idx >> 16) & 0xFF;
        ui_g_now_strings[i].operate_tpyel = 1;
#ifndef MANUAL_DIRTY
        ui_g_last_strings[i] = ui_g_now_strings[i];
#endif
        ui_g_dirty_string[i] = 1;
        idx++;
    }

    ui_scan_and_send(ui_g_now_figures, ui_g_dirty_figure, ui_g_now_strings, ui_g_dirty_string, TOTAL_FIGURE, TOTAL_STRING);

    for (int i = 0; i < TOTAL_FIGURE; i++) {
        ui_g_now_figures[i].operate_tpyel = 2;
    }
    for (int i = 0; i < TOTAL_STRING; i++) {
        ui_g_now_strings[i].operate_tpyel = 2;
    }
}

void ui_update_g() {
#ifndef MANUAL_DIRTY
    for (int i = 0; i < TOTAL_FIGURE; i++) {
        if (memcmp(&ui_g_now_figures[i], &ui_g_last_figures[i], sizeof(ui_g_now_figures[i])) != 0) {
            ui_g_dirty_figure[i] = 1;
            ui_g_last_figures[i] = ui_g_now_figures[i];
        }
    }
    for (int i = 0; i < TOTAL_STRING; i++) {
        if (memcmp(&ui_g_now_strings[i], &ui_g_last_strings[i], sizeof(ui_g_now_strings[i])) != 0) {
            ui_g_dirty_string[i] = 1;
            ui_g_last_strings[i] = ui_g_now_strings[i];
        }
    }
#endif
    ui_scan_and_send(ui_g_now_figures, ui_g_dirty_figure, ui_g_now_strings, ui_g_dirty_string, TOTAL_FIGURE, TOTAL_STRING);
}
