//
// Created by RM UI Designer
// Dynamic Edition
//

#ifndef UI_g_H
#define UI_g_H

#include "ui_interface.h"

extern ui_interface_figure_t ui_g_now_figures[11];
extern ui_interface_string_t ui_g_now_strings[5];
extern uint8_t ui_g_dirty_figure[11];
extern uint8_t ui_g_dirty_string[5];

#define ui_g_Ungroup_HEIGHT_LINE ((ui_interface_line_t*)&(ui_g_now_figures[0]))
#define ui_g_Ungroup_ARROW1 ((ui_interface_line_t*)&(ui_g_now_figures[1]))
#define ui_g_Ungroup_ARROW2 ((ui_interface_line_t*)&(ui_g_now_figures[2]))
#define ui_g_Ungroup_heigth_index ((ui_interface_round_t*)&(ui_g_now_figures[3]))
#define ui_g_Ungroup_REAL_HEIGHT ((ui_interface_number_t*)&(ui_g_now_figures[4]))
#define ui_g_Ungroup_SOURCE_LINE1 ((ui_interface_line_t*)&(ui_g_now_figures[5]))
#define ui_g_Ungroup_SOURCE_LINE2 ((ui_interface_line_t*)&(ui_g_now_figures[6]))
#define ui_g_Ungroup_SILVER1_LOCATION ((ui_interface_line_t*)&(ui_g_now_figures[7]))
#define ui_g_Ungroup_SILVER2_LOCATION ((ui_interface_line_t*)&(ui_g_now_figures[8]))
#define ui_g_Ungroup_NewRound ((ui_interface_round_t*)&(ui_g_now_figures[9]))
#define ui_g_Ungroup_SOURCE_HEIGHT ((ui_interface_line_t*)&(ui_g_now_figures[10]))

#define ui_g_Ungroup_SUCKER (&(ui_g_now_strings[0]))
#define ui_g_Ungroup_SUCKER_STATE (&(ui_g_now_strings[1]))
#define ui_g_Ungroup_ROBOT (&(ui_g_now_strings[2]))
#define ui_g_Ungroup_ROBOT_STATE (&(ui_g_now_strings[3]))
#define ui_g_Ungroup_HEIGHT_NAME (&(ui_g_now_strings[4]))

#ifdef MANUAL_DIRTY
#define ui_g_Ungroup_HEIGHT_LINE_dirty (ui_g_dirty_figure[0])
#define ui_g_Ungroup_ARROW1_dirty (ui_g_dirty_figure[1])
#define ui_g_Ungroup_ARROW2_dirty (ui_g_dirty_figure[2])
#define ui_g_Ungroup_heigth_index_dirty (ui_g_dirty_figure[3])
#define ui_g_Ungroup_REAL_HEIGHT_dirty (ui_g_dirty_figure[4])
#define ui_g_Ungroup_SOURCE_LINE1_dirty (ui_g_dirty_figure[5])
#define ui_g_Ungroup_SOURCE_LINE2_dirty (ui_g_dirty_figure[6])
#define ui_g_Ungroup_SILVER1_LOCATION_dirty (ui_g_dirty_figure[7])
#define ui_g_Ungroup_SILVER2_LOCATION_dirty (ui_g_dirty_figure[8])
#define ui_g_Ungroup_NewRound_dirty (ui_g_dirty_figure[9])
#define ui_g_Ungroup_SOURCE_HEIGHT_dirty (ui_g_dirty_figure[10])

#define ui_g_Ungroup_SUCKER_dirty (ui_g_dirty_string[0])
#define ui_g_Ungroup_SUCKER_STATE_dirty (ui_g_dirty_string[1])
#define ui_g_Ungroup_ROBOT_dirty (ui_g_dirty_string[2])
#define ui_g_Ungroup_ROBOT_STATE_dirty (ui_g_dirty_string[3])
#define ui_g_Ungroup_HEIGHT_NAME_dirty (ui_g_dirty_string[4])
#endif

void ui_init_g();
void ui_update_g();

#endif //UI_g_H
