import type { NotificationPermission, Notifier, Reminder } from '@vova/device';
import * as Notifications from 'expo-notifications';

// Alerts raised by a scan arrive while the app is open, so show them as banners then too.
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowBanner: true,
    shouldShowList: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
  }),
});

const REMINDER_PREFIX = 'reminder-';

function toPermission(status: Notifications.NotificationPermissionsStatus): NotificationPermission {
  if (status.granted) return 'granted';
  if (status.ios?.status === Notifications.IosAuthorizationStatus.PROVISIONAL) return 'granted';
  return status.canAskAgain ? 'undetermined' : 'denied';
}

/** Local notifications only — there is no push server. */
export const phoneNotifier: Notifier = {
  async permission() {
    return toPermission(await Notifications.getPermissionsAsync());
  },
  async request() {
    return toPermission(
      await Notifications.requestPermissionsAsync({
        ios: { allowAlert: true, allowSound: true, allowBadge: false },
      }),
    );
  },
  async notify(title, body) {
    await Notifications.scheduleNotificationAsync({ content: { title, body }, trigger: null });
  },
  async scheduleReminders(reminders: Reminder[]) {
    const scheduled = await Notifications.getAllScheduledNotificationsAsync();
    for (const item of scheduled) {
      if (item.identifier.startsWith(REMINDER_PREFIX)) {
        await Notifications.cancelScheduledNotificationAsync(item.identifier);
      }
    }
    for (const reminder of reminders) {
      await Notifications.scheduleNotificationAsync({
        identifier: reminder.id,
        content: { title: reminder.title, body: reminder.body },
        trigger: { type: Notifications.SchedulableTriggerInputTypes.DATE, date: reminder.at },
      });
    }
  },
};
